import os
import time
import psutil
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import nibabel as nib
import SimpleITK as sitk
import numpy as np
import pandas as pd
from scipy.ndimage import zoom
from collections import OrderedDict
import sys
sys.path.append("../3D-CNN-PyTorch")
from models import cnn

LABEL_MAP = {
    'Good': 1,
    'Acceptable': 0,
    'Poor': 0,
    'Missed': 0
}

# ---------- Dataset ----------
class MRIDataset(Dataset):
    def __init__(self, image_dir, mask_dir, split_csv, label_csv, split='train', target_shape=(128, 128, 128)):
        print(f"[INFO] Initializing MRIDataset with split='{split}'...")
        self.samples = []
        self.labels = {}
        self.target_shape = target_shape

        split_df = pd.read_csv(split_csv)
        if split == 'train':
            patient_ids = split_df['train_split'].dropna().values[:100]
        elif split == 'test':
            patient_ids = split_df['test_split'].dropna().values[:20]
        else:
            raise ValueError("[ERROR] split must be 'train' or 'test'")
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

        print(f"[INFO] Loaded {len(self.samples)} valid samples from disk.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        mri_path, mask_path, pid = self.samples[idx]
        tensor = preprocess(mri_path, mask_path, self.target_shape)
        label = torch.tensor([self.labels[pid]], dtype=torch.float32)
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
    print(f"[PREPROCESS] Loading MRI from: {mri_path}")
    mri_np, mri_spacing = load_nifti(mri_path)
    print(f"[PREPROCESS] Loading Mask from: {mask_path}")
    mask_np, _ = load_nifti(mask_path)

    print("[PREPROCESS] Running N4 bias correction...")
    mri_np = n4_bias_correction(mri_np)

    print("[PREPROCESS] Denoising MRI...")
    mri_np = denoise_image(mri_np)

    print("[PREPROCESS] Resampling to isotropic voxel spacing...")
    mri_np = resample_image(mri_np, mri_spacing)
    mask_np = resample_image(mask_np, mri_spacing)

    print("[PREPROCESS] Resizing to target shape...")
    mri_np = zoom(mri_np, np.array(target_shape) / np.array(mri_np.shape), order=1)
    mask_np = zoom(mask_np, np.array(target_shape) / np.array(mask_np.shape), order=0)
    mask_np = (mask_np > 0).astype(np.float32)

    stacked = np.stack([mri_np, mask_np], axis=0)
    return torch.tensor(stacked, dtype=torch.float32).unsqueeze(0)

# ---------- CNN ----------
class cnn3d(nn.Module):
    def __init__(self):
        super(cnn3d, self).__init__()

        self.conv1 = self._conv_layer_set(2, 16)  # 2 input channels (e.g., MRI + mask)
        self.conv2 = self._conv_layer_set(16, 32)
        self.conv3 = self._conv_layer_set(32, 64)

        self.conv1_bn = nn.BatchNorm3d(16)
        self.conv2_bn = nn.BatchNorm3d(32)
        self.conv3_bn = nn.BatchNorm3d(64)

        # Use adaptive pooling to avoid hardcoding input shape
        self.global_pool = nn.AdaptiveAvgPool3d((1, 1, 1))

        self.fc1 = nn.Linear(64, 128)
        self.fc1_bn = nn.BatchNorm1d(128)
        self.drop = nn.Dropout(p=0.3)
        self.fc2 = nn.Linear(128, 1)  # Binary classification

        self.relu = nn.LeakyReLU()

    def _conv_layer_set(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv1_bn(x)

        x = self.conv2(x)
        x = self.conv2_bn(x)

        x = self.conv3(x)
        x = self.conv3_bn(x)

        x = self.global_pool(x)             # [B, 64, 1, 1, 1]
        x = x.view(x.size(0), -1)           # [B, 64]

        x = self.fc1(x)                     # [B, 128]
        x = self.relu(x)
        x = self.fc1_bn(x)
        x = self.drop(x)

        x = self.fc2(x)                     # [B, 1]
        return x                            # Logits for BCEWithLogitsLoss

def generate_model():
    return cnn.cnn3d()

# ---------- Single-GPU Training ----------
def train_single_gpu():
    print("[INFO] Starting single-GPU training...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    dataset = MRIDataset("../images", "../segmentations/automatic",
                         "../train_test_splits.csv", "../preliminary_automatic_segmentations_quality_scores.csv",
                         split="train")

    dataloader = DataLoader(dataset, batch_size=128, shuffle=True, num_workers=4, pin_memory=True)
    print(f"[INFO] DataLoader initialized with {len(dataloader)} batches.")

    model = generate_model().to(device)
    print("[INFO] Model loaded and moved to device.")

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.BCEWithLogitsLoss()

    torch.backends.cudnn.benchmark = True
    torch.cuda.reset_peak_memory_stats()
    start_time = time.time()

    model.train()
    print("[INFO] Training loop started...")
    for i, (inputs, labels) in enumerate(dataloader):
        batch_start = time.time()
        inputs = inputs.to(device)
        labels = labels.to(device).view(-1)
        outputs = model(inputs).view(-1)

        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        batch_end = time.time()

        if i % 1 == 0:
            print(f"[BATCH {i:03d}] Loss: {loss.item():.4f} | Time: {batch_end - batch_start:.2f}s")

    end_time = time.time()
    peak_mem = torch.cuda.max_memory_allocated() / 1e6
    cpu_mem = psutil.Process(os.getpid()).memory_info().rss / 1e6

    print("[INFO] Training completed.")
    print(f"[INFO] Total Time: {end_time - start_time:.2f}s | Peak GPU Mem: {peak_mem:.2f}MB | CPU Mem: {cpu_mem:.2f}MB")

    os.makedirs("logs", exist_ok=True)
    with open("logs/single_gpu_resource_report.txt", "w") as f:
        f.write("=== Single GPU Resource Usage Report ===\n")
        f.write(f"Batch size: 128\n")
        f.write(f"Training samples: {len(dataset)}\n")
        f.write(f"Training time: {end_time - start_time:.2f} seconds\n")
        f.write(f"Peak GPU memory: {peak_mem:.2f} MB\n")
        f.write(f"CPU memory usage (RSS): {cpu_mem:.2f} MB\n")

    os.makedirs("checkpoints", exist_ok=True)
    torch.save(model.state_dict(), "checkpoints/single_gpu_epoch1.pth")
    print("[INFO] Model checkpoint saved.")

# ---------- Entry Point ----------
if __name__ == "__main__":
    train_single_gpu()
