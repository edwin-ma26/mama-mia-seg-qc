import os
import time
import psutil
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import nibabel as nib
import SimpleITK as sitk
import numpy as np
import pandas as pd
from scipy.ndimage import zoom
from collections import OrderedDict

LABEL_MAP = {
    'Good': 1,
    'Acceptable': 0,
    'Poor': 0,
    'Missed': 0
}

# ---------- Dataset ----------
class MRIDataset(Dataset):
    def __init__(self, image_dir, mask_dir, split_csv, label_csv, split='train', target_shape=(128, 128, 128)):
        print("Loading dataset...")
        self.samples = []
        self.labels = {}
        self.target_shape = target_shape

        split_df = pd.read_csv(split_csv)
        if split == 'train':
            patient_ids = split_df['train_split'].dropna().values[:150]
        elif split == 'test':
            patient_ids = split_df['test_split'].dropna().values[:30]
        else:
            raise ValueError("split must be 'train' or 'test'")
        patient_ids = set(patient_ids)

        label_df = pd.read_csv(label_csv)
        for _, row in label_df.iterrows():
            pid = row['patient_id']
            exp1, exp2 = row['expert_1_qs'], row['expert_2_qs']
            self.labels[pid] = 1 if exp1 == 'Good' and exp2 == 'Good' else 0

        for patient_id in patient_ids:
            image_path = os.path.join(image_dir, patient_id, f"{patient_id}_0000.nii.gz")
            mask_path = os.path.join(mask_dir, f"{patient_id}.nii.gz")
            if os.path.exists(image_path) and os.path.exists(mask_path) and patient_id in self.labels:
                self.samples.append((image_path, mask_path, patient_id))

        print(f"Loaded {len(self.samples)} samples.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        mri_path, mask_path, pid = self.samples[idx]
        tensor = preprocess(mri_path, mask_path, self.target_shape)
        label = torch.tensor([self.labels[pid]], dtype=torch.long)
        return tensor.squeeze(0), label

# ---------- Preprocessing ----------
def load_nifti(path):
    img = nib.load(path)
    return img.get_fdata(), img.header.get_zooms()

def n4_bias_correction(image_np):
    sitk_image = sitk.GetImageFromArray(image_np.astype(np.float32))
    mask_image = sitk.OtsuThreshold(sitk_image, 0, 1, 200)
    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    corrected = corrector.Execute(sitk_image, mask_image)
    return sitk.GetArrayFromImage(corrected)

def denoise_image(image_np):
    sitk_image = sitk.GetImageFromArray(image_np.astype(np.float32))
    denoised = sitk.CurvatureFlow(image1=sitk_image, timeStep=0.125, numberOfIterations=5)
    return sitk.GetArrayFromImage(denoised)

def resample_image(img_np, original_spacing, new_spacing=[1.0, 1.0, 1.0]):
    resize_factor = np.array(original_spacing) / np.array(new_spacing)
    new_shape = np.round(np.array(img_np.shape) * resize_factor).astype(int)
    zoom_factors = new_shape / np.array(img_np.shape)
    return zoom(img_np, zoom_factors, order=1)

def preprocess(mri_path, mask_path, target_shape=(128, 128, 128)):
    mri_np, mri_spacing = load_nifti(mri_path)
    mask_np, _ = load_nifti(mask_path)
    mri_np = n4_bias_correction(mri_np)
    mri_np = denoise_image(mri_np)
    mri_np = resample_image(mri_np, mri_spacing)
    mask_np = resample_image(mask_np, mri_spacing)
    mri_np = zoom(mri_np, np.array(target_shape) / np.array(mri_np.shape), order=1)
    mask_np = zoom(mask_np, np.array(target_shape) / np.array(mask_np.shape), order=0)
    mask_np = (mask_np > 0).astype(np.float32)
    stacked = np.stack([mri_np, mask_np], axis=0)
    return torch.tensor(stacked, dtype=torch.float32).unsqueeze(0)

# ---------- DenseNet ----------
class _DenseLayer(nn.Sequential):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate):
        super().__init__()
        self.add_module('norm1', nn.BatchNorm3d(num_input_features))
        self.add_module('relu1', nn.ReLU(inplace=True))
        self.add_module('conv1', nn.Conv3d(num_input_features, bn_size * growth_rate, 1, stride=1, bias=False))
        self.add_module('norm2', nn.BatchNorm3d(bn_size * growth_rate))
        self.add_module('relu2', nn.ReLU(inplace=True))
        self.add_module('conv2', nn.Conv3d(bn_size * growth_rate, growth_rate, 3, stride=1, padding=1, bias=False))
        self.drop_rate = drop_rate

    def forward(self, x):
        new_features = super().forward(x)
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return torch.cat([x, new_features], 1)

class _DenseBlock(nn.Sequential):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate):
        super().__init__()
        for i in range(num_layers):
            layer = _DenseLayer(num_input_features + i * growth_rate, growth_rate, bn_size, drop_rate)
            self.add_module(f'denselayer{i+1}', layer)

class _Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super().__init__()
        self.add_module('norm', nn.BatchNorm3d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv3d(num_input_features, num_output_features, 1, stride=1, bias=False))
        self.add_module('pool', nn.AvgPool3d(2, stride=2))

class DenseNet(nn.Module):
    def __init__(self, n_input_channels=2, conv1_t_size=7, conv1_t_stride=1, no_max_pool=False,
                 growth_rate=32, block_config=(6, 12, 24, 16), num_init_features=64,
                 bn_size=4, drop_rate=0, num_classes=2):
        super().__init__()
        self.features = [('conv1', nn.Conv3d(n_input_channels, num_init_features, (conv1_t_size, 7, 7),
                                             stride=(conv1_t_stride, 2, 2), padding=(conv1_t_size // 2, 3, 3), bias=False)),
                         ('norm1', nn.BatchNorm3d(num_init_features)),
                         ('relu1', nn.ReLU(inplace=True))]
        if not no_max_pool:
            self.features.append(('pool1', nn.MaxPool3d(3, stride=2, padding=1)))
        self.features = nn.Sequential(OrderedDict(self.features))

        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(num_layers, num_features, bn_size, growth_rate, drop_rate)
            self.features.add_module(f'denseblock{i+1}', block)
            num_features += num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition(num_features, num_features // 2)
                self.features.add_module(f'transition{i+1}', trans)
                num_features = num_features // 2

        self.features.add_module('norm5', nn.BatchNorm3d(num_features))
        self.classifier = nn.Linear(num_features, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        features = self.features(x)
        out = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool3d(out, (1, 1, 1)).view(features.size(0), -1)
        return self.classifier(out)

def generate_model(depth):
    return DenseNet()

# ---------- DDP Training Function ----------
def train(rank, world_size):
    dist.init_process_group(backend="nccl", init_method="env://", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

    print("Instantiating dataset...")
    dataset = MRIDataset("images", "segmentations/automatic",
                         "train_test_splits.csv", "preliminary_automatic_segmentations_quality_scores.csv",
                         split="train")
    
    print("Creating sampler and dataloader...")
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
    dataloader = DataLoader(dataset, batch_size=128, sampler=sampler, num_workers=4, pin_memory=True)

    print("Dataloader created. Entering training loop.")
    model = generate_model(121).to(rank)
    model = DDP(model, device_ids=[rank])
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()

    torch.cuda.reset_peak_memory_stats(rank)
    start_time = time.time()

    model.train()
    for i, (inputs, labels) in enumerate(dataloader):
        print("First batch loaded")
        inputs, labels = inputs.to(rank), labels.to(rank).squeeze()
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        if rank == 0 and i % 10 == 0:
            print(f"Processed batch {i}/{len(dataloader)}")

    end_time = time.time()
    peak_mem = torch.cuda.max_memory_allocated(rank) / 1e6
    process = psutil.Process(os.getpid())
    cpu_mem = process.memory_info().rss / 1e6

    if rank == 0:
        print("Starting training...")
        os.makedirs("logs", exist_ok=True)
        with open("logs/ddp_resource_report.txt", "w") as f:
            f.write("=== DDP Resource Usage Report ===\n")
            f.write(f"World size (GPUs): {world_size}\n")
            f.write(f"Batch size: 128\n")
            f.write(f"Training samples: {len(dataset)}\n")
            f.write(f"Training time: {end_time - start_time:.2f} seconds\n")
            f.write(f"Peak GPU memory (rank 0): {peak_mem:.2f} MB\n")
            f.write(f"CPU memory usage (RSS): {cpu_mem:.2f} MB\n")

        os.makedirs("checkpoints", exist_ok=True)
        torch.save(model.module.state_dict(), "checkpoints/densenet_epoch1.pth")

    dist.destroy_process_group()

# ---------- DDP Launcher ----------
if __name__ == "__main__":
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    world_size = 4
    mp.spawn(train, args=(world_size,), nprocs=world_size, join=True)
