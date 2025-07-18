import os
import numpy as np
import pandas as pd
import SimpleITK as sitk
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from tqdm import tqdm
import json
import argparse

# ---------- Config ----------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 50
IMG_SIZE = (384, 384)
OUTPUT_DIR = "./output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

EARLY_STOPPING_PATIENCE = 7

# ---------- Model ----------
class CNNBackbone(nn.Module):
    def __init__(self, img_size=IMG_SIZE, attn_dropout=0.1):
        super(CNNBackbone, self).__init__()
        self.conv1 = nn.Conv2d(2, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.dropout = nn.Dropout(0.1)
        self.fc1 = nn.Linear(64 * (img_size[0] // 4) * (img_size[1] // 4), 64)
        self.slice_classifier = nn.Linear(64, 1)
        self.slice_attn = nn.Linear(64, 1)
        self.attn_dropout = nn.Dropout(attn_dropout)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        logit = self.slice_classifier(x).squeeze(-1)
        attn_weight = self.slice_attn(x).squeeze(-1)
        attn_weight = self.attn_dropout(attn_weight)
        return logit, attn_weight

# ---------- Dataset ----------
class AllSlicesDataset(Dataset):
    def __init__(self, patient_boundaries, img_size=IMG_SIZE):
        self.samples = []
        qs_df = pd.read_csv("../preliminary_automatic_segmentations_quality_scores.csv", dtype=str)
        qs_df = qs_df.set_index("patient_id")
        for pid, start_slice, end_slice in patient_boundaries:
            if pid not in qs_df.index:
                continue
            expert_1 = qs_df.loc[pid, "expert_1_qs"]
            expert_2 = qs_df.loc[pid, "expert_2_qs"]
            label = 0 if (expert_1 == "Good" and expert_2 == "Good") else 1
            data_sources = []
            img_path = f"../preprocessed/{pid}/{pid}_n4_denoised_resampled_padded.nii.gz"
            mask_path = f"../preprocessed/{pid}/{pid}_mask_resampled_padded.nii.gz"
            if os.path.exists(img_path) and os.path.exists(mask_path):
                data_sources.append((img_path, mask_path, "original"))
            for aug_num in ["01", "02", "03"]:
                aug_img_path = f"../preprocessed/{pid}/{pid}_n4_denoised_resampled_padded_aug_{aug_num}.nii.gz"
                aug_mask_path = f"../preprocessed/{pid}/{pid}_mask_resampled_padded_aug_{aug_num}.nii.gz"
                if os.path.exists(aug_img_path) and os.path.exists(aug_mask_path):
                    data_sources.append((aug_img_path, aug_mask_path, f"aug_{aug_num}"))
            for img_path, mask_path, source_type in data_sources:
                img = sitk.GetArrayFromImage(sitk.ReadImage(img_path))
                mask = sitk.GetArrayFromImage(sitk.ReadImage(mask_path))
                slices = []
                for i in range(start_slice, end_slice + 1):
                    if i >= mask.shape[0] or mask[i, :, :].sum() == 0:
                        continue
                    img_slice = img[i, :, :]
                    mask_slice = mask[i, :, :]
                    slice_2ch = np.stack([img_slice, mask_slice], axis=0)
                    slice_tensor = torch.tensor(slice_2ch, dtype=torch.float32)
                    slices.append(slice_tensor)
                if len(slices) > 0:
                    patient_tensor = torch.stack(slices)
                    self.samples.append((patient_tensor, torch.tensor(label, dtype=torch.float32)))
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        return self.samples[idx]

# ---------- Train/Test Split ----------
def load_patient_boundaries(file_path):
    boundaries = []
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 3:
                patient_id = parts[0]
                start_slice = int(parts[1])
                end_slice = int(parts[2])
                boundaries.append((patient_id, start_slice, end_slice))
    return boundaries

def train_model_with_params(entropy_reg_weight, attn_dropout, weight_decay, temperature, train_loader, test_loader, param_combo_idx):
    model = CNNBackbone(attn_dropout=attn_dropout).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=weight_decay)
    criterion = nn.BCEWithLogitsLoss()
    training_logs = []
    best_val_f1 = -1
    best_epoch = -1
    epochs_no_improve = 0
    best_model_state = None
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        train_preds, train_labels = [], []
        train_logits = []
        for patient_slices, patient_label in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            patient_slices = patient_slices.squeeze(0).to(DEVICE)
            patient_label = patient_label.to(DEVICE)
            optimizer.zero_grad()
            slice_logits, slice_attn = model(patient_slices)
            weights = torch.softmax(slice_attn, dim=0)
            volume_logit = torch.sum(weights * slice_logits)
            scaled_logit = volume_logit / temperature
            classification_loss = criterion(scaled_logit.unsqueeze(0), patient_label)
            entropy = -torch.sum(weights * torch.log(weights + 1e-8))
            entropy_loss = -entropy_reg_weight * entropy
            total_loss = classification_loss + entropy_loss
            total_loss.backward()
            optimizer.step()
            running_loss += total_loss.item()
            pred = torch.sigmoid(scaled_logit).item()
            train_preds.append(1 if pred > 0.5 else 0)
            train_labels.append(int(patient_label.item()))
            train_logits.append(scaled_logit.item())
        avg_train_loss = running_loss / len(train_loader)
        train_acc = np.mean(np.array(train_preds) == np.array(train_labels))
        try:
            train_auc = roc_auc_score(train_labels, train_logits)
        except:
            train_auc = 0.5
        model.eval()
        preds, labels = [], []
        val_logits = []
        val_running_loss = 0.0
        with torch.no_grad():
            for patient_slices, patient_label in test_loader:
                patient_slices = patient_slices.squeeze(0).to(DEVICE)
                patient_label = patient_label.item()
                slice_logits, slice_attn = model(patient_slices)
                weights = torch.softmax(slice_attn, dim=0)
                volume_logit = torch.sum(weights * slice_logits)
                scaled_logit = volume_logit / temperature
                pred = torch.sigmoid(scaled_logit).item()
                preds.append(1 if pred > 0.5 else 0)
                labels.append(int(patient_label))
                val_logits.append(scaled_logit.item())
                val_loss = criterion(scaled_logit.unsqueeze(0), torch.tensor([patient_label], device=DEVICE, dtype=torch.float32))
                val_running_loss += val_loss.item()
        acc = np.mean(np.array(preds) == np.array(labels))
        avg_val_loss = val_running_loss / len(test_loader)
        try:
            precision = precision_score(labels, preds)
        except:
            precision = 0.0
        try:
            recall = recall_score(labels, preds)
        except:
            recall = 0.0
        try:
            f1 = f1_score(labels, preds)
        except:
            f1 = 0.0
        try:
            auc = roc_auc_score(labels, val_logits)
        except:
            auc = 0.5
        training_logs.append({
            "epoch": epoch + 1,
            "train_loss": avg_train_loss,
            "train_accuracy": train_acc,
            "train_auc": train_auc,
            "val_loss": avg_val_loss,
            "val_accuracy": acc,
            "val_precision": precision,
            "val_recall": recall,
            "val_f1": f1,
            "val_auc": auc,
            "train_logits_mean": np.mean(train_logits),
            "train_logits_std": np.std(train_logits),
            "val_logits_mean": np.mean(val_logits),
            "val_logits_std": np.std(val_logits)
        })
        if f1 > best_val_f1:
            best_val_f1 = f1
            best_epoch = epoch
            epochs_no_improve = 0
            best_model_state = model.state_dict()
        else:
            epochs_no_improve += 1
        if epochs_no_improve >= EARLY_STOPPING_PATIENCE:
            print(f"Early stopping at epoch {epoch+1} (no improvement in val_f1 for {EARLY_STOPPING_PATIENCE} epochs). Best val_f1: {best_val_f1:.4f} at epoch {best_epoch+1}", flush=True)
            break
    return {
        'best_val_f1': best_val_f1,
        'best_epoch': best_epoch,
        'training_logs': training_logs,
        'best_model_state': best_model_state,
        'final_epoch': epoch + 1
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--entropy_reg_weight', type=float, required=True)
    parser.add_argument('--attn_dropout', type=float, required=True)
    parser.add_argument('--weight_decay', type=float, required=True)
    parser.add_argument('--temperature', type=float, required=True)
    parser.add_argument('--combo_idx', type=int, required=True)
    args = parser.parse_args()

    train_file = '../filter/aggregated/train_patients_aggregated.txt'
    test_file = '../filter/aggregated/test_patients_aggregated.txt'
    train_boundaries = load_patient_boundaries(train_file)
    test_boundaries = load_patient_boundaries(test_file)
    print(f"Loading training dataset with {len(train_boundaries)} patient boundaries...")
    train_dataset = AllSlicesDataset(train_boundaries)
    print(f"Training dataset loaded: {len(train_dataset)} samples")
    print(f"Loading test dataset with {len(test_boundaries)} patient boundaries...")
    test_dataset = AllSlicesDataset(test_boundaries)
    print(f"Test dataset loaded: {len(test_dataset)} samples")
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    result = train_model_with_params(
        entropy_reg_weight=args.entropy_reg_weight,
        attn_dropout=args.attn_dropout,
        weight_decay=args.weight_decay,
        temperature=args.temperature,
        train_loader=train_loader,
        test_loader=test_loader,
        param_combo_idx=args.combo_idx
    )
    combo_dir = os.path.join(OUTPUT_DIR, f"param_combo_{args.combo_idx+1:03d}")
    os.makedirs(combo_dir, exist_ok=True)
    with open(os.path.join(combo_dir, "training_log.json"), "w") as f:
        json.dump(result['training_logs'], f, indent=4)
    pd.DataFrame(result['training_logs']).to_csv(os.path.join(combo_dir, "training_log.csv"), index=False)
    if result['best_model_state'] is not None:
        torch.save(result['best_model_state'], os.path.join(combo_dir, "best_model.pt"))
    config = {
        'entropy_reg_weight': args.entropy_reg_weight,
        'attn_dropout': args.attn_dropout,
        'weight_decay': args.weight_decay,
        'temperature': args.temperature,
        'best_val_f1': result['best_val_f1'],
        'best_epoch': result['best_epoch'],
        'final_epoch': result['final_epoch']
    }
    with open(os.path.join(combo_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=4)
    print(f"Completed combo {args.combo_idx+1}. Best val_f1: {result['best_val_f1']:.4f}")

if __name__ == "__main__":
    main() 