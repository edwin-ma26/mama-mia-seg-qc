import os
import numpy as np
import nibabel as nib
import SimpleITK as sitk
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import zoom
from collections import OrderedDict

# ---------- DenseNet Definition ----------
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

    def forward(self, x):
        features = self.features(x)
        out = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool3d(out, output_size=(1, 1, 1)).view(features.size(0), -1)
        out = self.classifier(out)
        return out

def generate_model(model_depth):
    return DenseNet(num_init_features=64, growth_rate=32, block_config=(6, 12, 24, 16),
                    n_input_channels=2, num_classes=2)

# ---------- Preprocessing Functions ----------
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

    mri_np = (mri_np - np.mean(mri_np)) / np.std(mri_np)
    mask_np = (mask_np > 0).astype(np.float32)

    stacked = np.stack([mri_np, mask_np], axis=0)
    return torch.tensor(stacked, dtype=torch.float32).unsqueeze(0)  # (1, 2, D, H, W)

# ---------- Run Inference ----------
mri_path = "images/DUKE_001/DUKE_001_0000.nii.gz"
mask_path = "segmentations/automatic/DUKE_001.nii.gz"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

input_tensor = preprocess(mri_path, mask_path).to(device)

model = generate_model(121).to(device)

# Dummy label (e.g. 1 = good, 0 = bad) for testing
label = torch.tensor([1], dtype=torch.long).to(device)

# Set to training mode
model.train()

# Define loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# Run training loop
epochs = 5
for epoch in range(epochs):
    optimizer.zero_grad()
    output = model(input_tensor)
    loss = criterion(output, label)
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

model.eval()

with torch.no_grad():
    output = model(input_tensor)
    prediction = torch.argmax(output, dim=1).item()
    print("Prediction:", prediction)
    print("GPU memory used: {:.2f} MB".format(torch.cuda.memory_allocated() / 1024 / 1024))
