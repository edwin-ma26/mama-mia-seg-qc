# Full CNN Training Pipeline with Logging and Model Saving

import os
import numpy as np
import pandas as pd
import nibabel as nib
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import SimpleITK as sitk
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
from collections import Counter
from tqdm import tqdm
import json

# ---------------- Configuration ---------------- #
IMG_SIZE = (128, 128)
DATA_DIR = 'images'
AUTO_SEG_DIR = 'segmentations/automatic'
CSV_PATH = 'preliminary_automatic_segmentations_quality_scores.csv'
SPLIT_PATH = 'train_test_splits.csv'
BATCH_SIZE = 8
EPOCHS = 5
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

OUTPUT_DIR = "training_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

LABEL_MAP = {
    'Good': 1,
    'Acceptable': 0,
    'Poor': 0,
    'Missed': 0
}

# ---------------- Preprocessing ---------------- #
def preprocess_image(image_path, denoise=True):
    image = sitk.ReadImage(image_path)
    if denoise:
        image = sitk.CurvatureFlow(image1=image, timeStep=0.125, numberOfIterations=5)
    return sitk.GetArrayFromImage(image)

def get_best_slice(mask):
    return np.argmax(np.sum(mask, axis=(0, 1)))

# ---------------- Dataset ---------------- #
class SegmentationQualityDataset(Dataset):
    def __init__(self, csv_path, image_root, mask_root, transform=None):
        self.df = pd.read_csv(csv_path)
        self.image_root = image_root
        self.mask_root = mask_root
        self.transform = transform
        self.samples = []

        for _, row in self.df.iterrows():
            pid = row['patient_id']
            label1 = LABEL_MAP.get(row['expert_1_qs'], None)
            label2 = LABEL_MAP.get(row['expert_2_qs'], None)
            if label1 is not None and label2 is not None:
                final_label = int((label1 + label2) >= 2)
                self.samples.append((pid, final_label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        pid, label = self.samples[idx]
        image_folder = os.path.join(self.image_root, pid)
        mask_path = os.path.join(self.mask_root, f"{pid}.nii.gz")
        if not os.path.exists(mask_path):
            raise FileNotFoundError(f"Expected mask file not found: {mask_path}")

        image_file = next((f for f in os.listdir(image_folder) if f.endswith('.nii') or f.endswith('.nii.gz')), None)
        if image_file is None:
            raise FileNotFoundError(f"No .nii or .nii.gz image found in {image_folder}")
        
        image_path = os.path.join(image_folder, image_file)

        img_vol = preprocess_image(image_path)
        mask_vol = nib.load(mask_path).get_fdata()

        best_slice = get_best_slice(mask_vol)
        img = img_vol[best_slice]
        mask = mask_vol[:, :, best_slice].astype(np.float32)

        img = torch.tensor((img - img.mean()) / (img.std() + 1e-8), dtype=torch.float32).unsqueeze(0)
        mask = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)
        stacked = torch.cat([img, mask], dim=0)

        if self.transform:
            stacked = self.transform(stacked)

        return stacked, torch.tensor(label, dtype=torch.float32)

def resize_tensor(tensor, size):
    return F.interpolate(tensor.unsqueeze(0), size=size, mode='bilinear', align_corners=False).squeeze(0)

resize_transform = lambda x: resize_tensor(x, IMG_SIZE)

# ---------------- Data Preparation ---------------- #
dataset = SegmentationQualityDataset(CSV_PATH, DATA_DIR, AUTO_SEG_DIR, transform=resize_transform)
split_df = pd.read_csv(SPLIT_PATH)
train_ids = split_df['train_split'].dropna().tolist()
val_ids = split_df['test_split'].dropna().tolist()
train_samples = [sample for sample in dataset.samples if sample[0] in train_ids]
val_samples = [sample for sample in dataset.samples if sample[0] in val_ids]

class SubsetDataset(Dataset):
    def __init__(self, full_dataset, subset_samples):
        self.full_dataset = full_dataset
        self.subset_samples = subset_samples
        self.index_map = {pid: idx for idx, (pid, _) in enumerate(full_dataset.samples)}

    def __len__(self):
        return len(self.subset_samples)

    def __getitem__(self, idx):
        pid, label = self.subset_samples[idx]
        return self.full_dataset[self.index_map[pid]]

train_dataset = SubsetDataset(dataset, train_samples)
val_dataset = SubsetDataset(dataset, val_samples)

train_labels = [label for _, label in train_samples]
label_counts = Counter(train_labels)
train_weights = [1.0 / label_counts[label] for _, label in train_samples]
train_sampler = WeightedRandomSampler(train_weights, num_samples=len(train_weights), replacement=True)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=train_sampler)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# ---------------- Model ---------------- #
class CNNClassifier(nn.Module):
    def __init__(self):
        super(CNNClassifier, self).__init__()
        self.conv1 = nn.Conv2d(2, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(64 * (IMG_SIZE[0] // 4) * (IMG_SIZE[1] // 4), 64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return torch.sigmoid(self.fc2(x)).squeeze()

model = CNNClassifier().to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.BCELoss()

# ---------------- Training Loop ---------------- #
training_logs = []

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    avg_train_loss = running_loss / len(train_loader)
    print(f"Train Loss: {avg_train_loss:.4f}")

    # Validation
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs)
            preds = (outputs > 0.5).float()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = np.mean(np.array(all_preds) == np.array(all_labels))
    f1 = f1_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    auc = roc_auc_score(all_labels, all_preds)

    print(f"Validation Accuracy: {accuracy:.4f}")
    print(f"F1 Score: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, AUC: {auc:.4f}")

    # Save model and metrics
    torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, f"model_epoch_{epoch+1}.pt"))

    training_logs.append({
        "epoch": epoch + 1,
        "train_loss": avg_train_loss,
        "accuracy": accuracy,
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "auc": auc
    })

    with open(os.path.join(OUTPUT_DIR, "training_log.json"), "w") as f:
        json.dump(training_logs, f, indent=4)

    pd.DataFrame(training_logs).to_csv(os.path.join(OUTPUT_DIR, "training_log.csv"), index=False)
