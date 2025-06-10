import os
import glob
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import nibabel as nib
import SimpleITK as sitk
import numpy as np
import pandas as pd
from scipy.ndimage import zoom
from collections import OrderedDict

# ---------- Dataset ----------
import pandas as pd

LABEL_MAP = {
    'Good': 1,
    'Acceptable': 0,
    'Poor': 0,
    'Missed': 0
}

class MRIDataset(Dataset):
    def __init__(self, image_dir, mask_dir, split_csv, label_csv, split='train', target_shape=(128, 128, 128)):
        self.samples = []
        self.labels = {}
        self.target_shape = target_shape

        # Load split info
        split_df = pd.read_csv(split_csv)
        if split == 'train':
            patient_ids = split_df['train_split'].dropna().values
        elif split == 'test':
            patient_ids = split_df['test_split'].dropna().values
        else:
            raise ValueError("split must be 'train' or 'test'")
        patient_ids = set(patient_ids)

        # Load label info
        label_df = pd.read_csv(label_csv)
        for _, row in label_df.iterrows():
            pid = row['patient_id']
            exp1, exp2 = row['expert_1_qs'], row['expert_2_qs']
            self.labels[pid] = 1 if exp1 == 'Good' and exp2 == 'Good' else 0

        # Collect valid image/mask pairs + labels
        for patient_id in patient_ids:
            image_path = os.path.join(image_dir, patient_id, f"{patient_id}_0000.nii.gz")
            mask_path = os.path.join(mask_dir, f"{patient_id}.nii.gz")
            if os.path.exists(image_path) and os.path.exists(mask_path) and patient_id in self.labels:
                self.samples.append((image_path, mask_path, patient_id))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        mri_path, mask_path, pid = self.samples[idx]
        tensor = preprocess(mri_path, mask_path, self.target_shape)
        label = torch.tensor([self.labels[pid]], dtype=torch.long)
        return tensor.squeeze(0), label

# ---------- DenseNet Model (same as before) ----------
class _DenseLayer(nn.Sequential):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate):
        super().__init__()
        self.add_module('norm1', nn.BatchNorm3d(num_input_features))
        self.add_module('relu1', nn.ReLU(inplace=True))
        self.add_module('conv1', nn.Conv3d(num_input_features, bn_size * growth_rate, kernel_size=1, stride=1, bias=False))
        self.add_module('norm2', nn.BatchNorm3d(bn_size * growth_rate))
        self.add_module('relu2', nn.ReLU(inplace=True))
        self.add_module('conv2', nn.Conv3d(bn_size * growth_rate, growth_rate, kernel_size=3, stride=1, padding=1, bias=False))
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
            self.add_module('denselayer{}'.format(i + 1), layer)

class _Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super().__init__()
        self.add_module('norm', nn.BatchNorm3d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv3d(num_input_features, num_output_features, kernel_size=1, stride=1, bias=False))
        self.add_module('pool', nn.AvgPool3d(kernel_size=2, stride=2))

class DenseNet(nn.Module):
    def __init__(self, n_input_channels=2, conv1_t_size=7, conv1_t_stride=1, no_max_pool=False,
                 growth_rate=32, block_config=(6, 12, 24, 16), num_init_features=64,
                 bn_size=4, drop_rate=0, num_classes=2):
        super().__init__()
        self.features = [('conv1', nn.Conv3d(n_input_channels, num_init_features, kernel_size=(conv1_t_size, 7, 7),
                                             stride=(conv1_t_stride, 2, 2), padding=(conv1_t_size // 2, 3, 3), bias=False)),
                         ('norm1', nn.BatchNorm3d(num_init_features)),
                         ('relu1', nn.ReLU(inplace=True))]
        if not no_max_pool:
            self.features.append(('pool1', nn.MaxPool3d(kernel_size=3, stride=2, padding=1)))
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
        out = F.adaptive_avg_pool3d(out, output_size=(1, 1, 1)).view(features.size(0), -1)
        out = self.classifier(out)
        return out

def generate_model(model_depth):
    return DenseNet(num_init_features=64, growth_rate=32, block_config=(6, 12, 24, 16),
                    n_input_channels=2, num_classes=2)

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
    img_resampled = zoom(img_np, zoom_factors, order=1)
    return img_resampled

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
    return torch.tensor(stacked, dtype=torch.float32).unsqueeze(0)  # (1, 2, D, H, W)

# ---------- DDP Training ----------
def train(rank, world_size):
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

    dataset = MRIDataset(
        "images",
        "segmentations/automatic",
        split_csv="train_test_splits.csv",
        label_csv="preliminary_automatic_segmentations_quality_scores.csv",
        split="train"
    )

    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
    dataloader = DataLoader(dataset, batch_size=128, sampler=sampler, num_workers=8, pin_memory=True)

    model = generate_model(121).to(rank)
    model = DDP(model, device_ids=[rank])

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    if rank == 0:
        log_file = open("training_log.txt", "a")

    for epoch in range(5):
        model.train()
        sampler.set_epoch(epoch)
        for batch in dataloader:
            inputs, labels = batch
            inputs, labels = inputs.to(rank), labels.to(rank).squeeze()
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            if rank == 0:
                # Save model checkpoint
                checkpoint_path = f"checkpoints/model_epoch_{epoch+1}.pt"
                os.makedirs("checkpoints", exist_ok=True)
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.module.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss.item()
                }, checkpoint_path)
            
                # Log loss to file
                log_file.write(f"Epoch {epoch + 1}, Loss: {loss.item():.4f}\n")
                log_file.flush()


    if rank == 0:
        log_file.close()

    dist.destroy_process_group()

if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    mp.spawn(train, args=(world_size,), nprocs=world_size, join=True)
    test_dataset = MRIDataset(
        "images",
        "segmentations/automatic",
        split_csv="train_test_splits.csv",
        label_csv="preliminary_automatic_segmentations_quality_scores.csv",
        split="test"
    )
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4)
