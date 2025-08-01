import os
import numpy as np
import pandas as pd
import SimpleITK as sitk
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import KFold
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from tqdm import tqdm
import json
import itertools
from datetime import datetime
from monai.transforms import RandFlipd, RandAffined, Compose, Rand3DElasticd, Rand2DElasticd
from monai.data import MetaTensor
import concurrent.futures
from skimage.transform import resize
import random

# ---------- Config ----------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 50
IMG_SIZE = (384, 384)
OUTPUT_DIR = "./output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Create parameter sweep output directory
SWEEP_OUTPUT_DIR = "./output/parameter_sweep"
os.makedirs(SWEEP_OUTPUT_DIR, exist_ok=True)

# Early stopping config
EARLY_STOPPING_PATIENCE = 7  # Number of epochs to wait for improvement

# Temperature scaling config
TEMPERATURE = 1.0  # Set to >1.0 for softer, <1.0 for sharper. Tune as needed.

# Entropy regularization config
ENTROPY_REG_WEIGHT = 0.0005  # Weight for entropy regularization (encourages uniform attention)

# Attention regularization config
ATTN_BIAS_WEIGHT = 0.01  # You can tune this

# ---------- Parameter Sweep Config ----------
# Set to True to run parameter sweep, False to run single training
RUN_PARAMETER_SWEEP = False

# Parameter combinations for sweep
PARAMETER_COMBINATIONS = {
    'weight_decay': [0.0, 0.0001, 0.001],  # L2 regularization
    'entropy_reg_weight': [0.0, 0.0005, 0.001],  # Entropy regularization
    'dropout_rate': [0.1, 0.2, 0.3],  # Dropout rate for main layers
    'attn_dropout_rate': [0.0, 0.1, 0.2],  # Dropout rate for attention
}

# Chunk configuration for running 1/9th of parameters at a time
CHUNK_SIZE = 9  # Total combinations / 9 = 81 / 9 = 9 combinations per chunk
CHUNK_INDEX = 8  # Set this to 0-8 to run different chunks (0-based indexing)
SWEEP_OUTPUT_DIR = "./output/parameter_sweep"  # Fixed output directory for all chunks

# ---------- Model ----------
class CNNBackbone(nn.Module):
    def __init__(self, img_size=IMG_SIZE, dropout_rate=0.1, attn_dropout=0.1):
        super(CNNBackbone, self).__init__()
        self.conv1 = nn.Conv2d(2, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.dropout = nn.Dropout(dropout_rate)

        # compute new feature map size after 3 poolings
        fmap_size = (img_size[0] // 8) * (img_size[1] // 8)

        self.fc1 = nn.Linear(128 * fmap_size, 128)
        self.slice_classifier = nn.Linear(128, 1)
        self.slice_attn = nn.Linear(128, 1)
        self.attn_dropout = nn.Dropout(attn_dropout)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        logit = self.slice_classifier(x).squeeze(-1)
        attn_weight = self.slice_attn(x).squeeze(-1)
        attn_weight = self.attn_dropout(attn_weight)
        return logit, attn_weight

# ---------- Dataset ----------
class AllSlicesDataset(Dataset):
    def __init__(self, patient_ids, img_size=IMG_SIZE, augment=False):
        self.patient_data = []
        self.augment = augment
        qs_df = pd.read_csv("../preliminary_automatic_segmentations_quality_scores.csv", dtype=str)
        qs_df = qs_df.set_index("patient_id")

        # Define MONAI augmentations (dictionary-based from test2.ipynb)
        self.augmentation_transforms = Compose([
            RandFlipd(keys=["image", "mask"], prob=0.5, spatial_axis=1),
            RandAffined(
                keys=["image", "mask"],
                prob=0.95,
                rotate_range=(np.deg2rad(30),),
                scale_range=(0.2,),
                translate_range=(20,),
                mode=("bilinear", "nearest"),
                padding_mode="reflection"
            ),
            Rand2DElasticd(
                keys=["image", "mask"],
                spatial_size=(384, 384),
                magnitude_range=(2, 5),
                prob=0.95,
                mode=("bilinear", "nearest"),
                padding_mode="reflection",
                spacing=(32, 32)
            )
        ])

        for pid in patient_ids:
            print(f"Processing patient {pid}...", flush=True)
            if pid not in qs_df.index:
                continue
            expert_1 = qs_df.loc[pid, "expert_1_qs"]
            expert_2 = qs_df.loc[pid, "expert_2_qs"]
            label = 0 if (expert_1 == "Good" and expert_2 == "Good") else 1

            # Try both _padded and _cropped, use whichever exists
            img_path_padded = f"../preprocessed/{pid}/{pid}_n4_denoised_resampled_padded.nii.gz"
            mask_path_padded = f"../preprocessed/{pid}/{pid}_mask_resampled_padded.nii.gz"
            img_path_cropped = f"../preprocessed/{pid}/{pid}_n4_denoised_resampled_cropped.nii.gz"
            mask_path_cropped = f"../preprocessed/{pid}/{pid}_mask_resampled_cropped.nii.gz"

            if os.path.exists(img_path_padded) and os.path.exists(mask_path_padded):
                img_path = img_path_padded
                mask_path = mask_path_padded
            elif os.path.exists(img_path_cropped) and os.path.exists(mask_path_cropped):
                img_path = img_path_cropped
                mask_path = mask_path_cropped
            else:
                # Will be skipped by the next check
                img_path = img_path_padded
                mask_path = mask_path_padded
            if not (os.path.exists(img_path) and os.path.exists(mask_path)):
                continue

            # Check image size and skip if larger than IMG_SIZE
            try:
                img = sitk.GetArrayFromImage(sitk.ReadImage(img_path))  # [Z, Y, X]
                if img.shape[1] > IMG_SIZE[0] or img.shape[2] > IMG_SIZE[1]:
                    print(f"Skipping patient {pid} due to image size {img.shape[1:]} > {IMG_SIZE}")
                    continue
            except Exception as e:
                print(f"Error reading image for patient {pid}: {e}")
                continue

            self.patient_data.append((pid, img_path, mask_path, label))

    def __len__(self):
        return len(self.patient_data)

    @staticmethod
    def process_slice(i, img, mask, augment_fn):
        img_slice = img[i]
        mask_slice = mask[i]
        
        # Resize both image and mask to 384x384
        img_slice_resized = resize(img_slice, (384, 384), order=1, preserve_range=True, anti_aliasing=True).astype(np.float32)
        mask_slice_resized = resize(mask_slice, (384, 384), order=0, preserve_range=True, anti_aliasing=False).astype(np.float32)
        
        # Add channel dimension → shape becomes (1, 384, 384)
        img_slice_resized = np.expand_dims(img_slice_resized, axis=0)
        mask_slice_resized = np.expand_dims(mask_slice_resized, axis=0)
        
        if augment_fn:
            # Use dict for compatibility, but do not use MetaTensor
            data_dict = {
                "image": img_slice_resized.copy(),
                "mask": mask_slice_resized.copy()
            }
            augmented = augment_fn(data_dict)
            img_slice_resized = augmented["image"]
            mask_slice_resized = augmented["mask"]
        
        # Remove channel dimension and stack as 2-channel tensor
        img_slice_final = img_slice_resized.squeeze(0)  # (384, 384)
        mask_slice_final = mask_slice_resized.squeeze(0)  # (384, 384)
        slice_2ch = np.stack([img_slice_final, mask_slice_final], axis=0)  # (2, 384, 384)
        
        return torch.tensor(slice_2ch, dtype=torch.float32)

    def __getitem__(self, idx):
        pid, img_path, mask_path, label = self.patient_data[idx]
        img = sitk.GetArrayFromImage(sitk.ReadImage(img_path))  # [Z, Y, X]
        mask = sitk.GetArrayFromImage(sitk.ReadImage(mask_path))

        # Sequentially process each slice (no threading)
        slices = [self.process_slice(i, img, mask, self.augmentation_transforms if self.augment else None)
                  for i in range(mask.shape[0])]

        if len(slices) == 0:
            slices.append(torch.zeros((2, *img.shape[1:]), dtype=torch.float32))

        return torch.stack(slices), torch.tensor(label, dtype=torch.float32), pid

# ---------- Train/Test Split ----------
def load_patient_ids(file_path):
    """Load patient IDs from file with format: one patient_id per line"""
    patient_ids = []
    with open(file_path, 'r') as f:
        for line in f:
            patient_id = line.strip()
            if patient_id:  # Skip empty lines
                patient_ids.append(patient_id)
    return patient_ids

# Calculate class weights for weighted loss
def calculate_class_weights(dataset):
    """Calculate class weights based on dataset distribution"""
    good_count = 0
    bad_count = 0

    for _, _, _, label in dataset.patient_data:
        if label == 0:  # Good
            good_count += 1
        else:  # Bad
            bad_count += 1

    total = good_count + bad_count
    print(f"Class distribution: {good_count} good ({good_count/total*100:.2f}%), {bad_count} bad ({bad_count/total*100:.2f}%)")

    # Calculate pos_weight for BCEWithLogitsLoss
    pos_weight = good_count / bad_count
    print(f"Using pos_weight = {pos_weight:.4f} for weighted loss")

    return pos_weight

train_file = '../filter/train_patients.txt'
test_file = '../filter/test_patients.txt'

train_patient_ids = load_patient_ids(train_file)
test_patient_ids = load_patient_ids(test_file)

print(f"Loading training dataset with {len(train_patient_ids)} patients...")
train_dataset = AllSlicesDataset(train_patient_ids, augment=True)
print(f"Training dataset loaded: {len(train_dataset)} samples")

print(f"Loading test dataset with {len(test_patient_ids)} patients...")
test_dataset = AllSlicesDataset(test_patient_ids, augment=False)
print(f"Test dataset loaded: {len(test_dataset)} samples")

train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=4)  # 1 patient per batch
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4)

# Calculate class weights for training set
pos_weight = calculate_class_weights(train_dataset)

def train_model_with_params(params, train_loader, test_loader, run_id, output_dir):
    """Train a model with specific regularization parameters"""
    
    # Create model with specified dropout rates
    model = CNNBackbone(
        dropout_rate=params['dropout_rate'],
        attn_dropout=params['attn_dropout_rate']
    ).to(DEVICE)
    
    # Create optimizer with specified weight decay
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=1e-4, 
        weight_decay=params['weight_decay']
    )
    # Use weighted loss to handle class imbalance
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, device=DEVICE))
    
    # Training logs
    training_logs = []
    best_val_f1 = -1
    best_epoch = -1
    epochs_no_improve = 0
    best_model_state = None

    # Create directory for 2d-aug model checkpoints
    in_training_aug_dir = os.path.join("./output", "2d-aug")
    os.makedirs(in_training_aug_dir, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"RUN {run_id} (Chunk {CHUNK_INDEX + 1}): weight_decay={params['weight_decay']}, "
          f"entropy_reg={params['entropy_reg_weight']}, "
          f"dropout={params['dropout_rate']}, "
          f"attn_dropout={params['attn_dropout_rate']}")
    print(f"{'='*60}")
    
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        train_preds, train_labels = [], []
        train_logits = []
        
        for patient_slices, patient_label, patient_id in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
            patient_slices = patient_slices.squeeze(0).to(DEVICE)
            patient_label = patient_label.to(DEVICE)

            optimizer.zero_grad()
            slice_logits, slice_attn = model(patient_slices)
            weights = torch.softmax(slice_attn, dim=0)
            volume_logit = torch.sum(weights * slice_logits)
            scaled_logit = volume_logit / TEMPERATURE

            # Calculate main classification loss
            classification_loss = criterion(scaled_logit.unsqueeze(0), patient_label)

            # Calculate entropy regularization loss
            entropy = -torch.sum(weights * torch.log(weights + 1e-8))
            entropy_loss = -params['entropy_reg_weight'] * entropy

            # --- ADD THIS BLOCK ---
            mask = patient_slices[:, 1, :, :]  # [num_slices, 384, 384]
            mask_area = mask.sum(dim=(1, 2))   # [num_slices]
            importance_weights = mask_area / (mask_area.sum() + 1e-8)  # [num_slices]
            # Clamp weights and importance_weights
            weights = torch.clamp(weights, min=1e-6, max=1.0)
            importance_weights = torch.clamp(importance_weights, min=1e-6, max=1.0)
            attn_reg_loss = F.kl_div(weights.log(), importance_weights, reduction='batchmean')
            # Or use mse_loss instead:
            # attn_reg_loss = F.mse_loss(weights, importance_weights)
            ATTN_BIAS_WEIGHT = 0.01  # Move to config if you want

            # Debug: print logits and attention weights (first 5)
            print("Slice logits:", slice_logits[:5].detach().cpu().numpy())
            print("Attention weights:", slice_attn[:5].detach().cpu().numpy())

            # Total loss
            total_loss = classification_loss + entropy_loss + ATTN_BIAS_WEIGHT * attn_reg_loss
            # --- END BLOCK ---

            total_loss.backward()
            optimizer.step()
            running_loss += total_loss.item()

            # For train accuracy and logits
            pred = torch.sigmoid(scaled_logit).item()
            train_preds.append(1 if pred > 0.5 else 0)
            train_labels.append(int(patient_label.item()))
            train_logits.append(scaled_logit.item())

        avg_train_loss = running_loss / len(train_loader)
        train_acc = np.mean(np.array(train_preds) == np.array(train_labels))
        
        # Calculate training metrics
        try:
            train_precision = precision_score(train_labels, train_preds)
        except:
            train_precision = 0.0
        try:
            train_recall = recall_score(train_labels, train_preds)
        except:
            train_recall = 0.0
        try:
            train_f1 = f1_score(train_labels, train_preds)
        except:
            train_f1 = 0.0
        try:
            train_auc = roc_auc_score(train_labels, train_logits)
        except:
            train_auc = 0.5

        # Validation
        model.eval()
        preds, labels = [], []
        val_logits = []
        val_running_loss = 0.0
        
        with torch.no_grad():
            for patient_slices, patient_label, patient_id in test_loader:
                patient_slices = patient_slices.squeeze(0).to(DEVICE)
                patient_label = patient_label.item()
                slice_logits, slice_attn = model(patient_slices)
                weights = torch.softmax(slice_attn, dim=0)
                volume_logit = torch.sum(weights * slice_logits)
                scaled_logit = volume_logit / TEMPERATURE
                pred = torch.sigmoid(scaled_logit).item()
                preds.append(1 if pred > 0.5 else 0)
                labels.append(int(patient_label))
                val_logits.append(scaled_logit.item())
                
                val_loss = criterion(scaled_logit.unsqueeze(0), torch.tensor([patient_label], device=DEVICE, dtype=torch.float32))
                val_running_loss += val_loss.item()


                if random.random() < 0.01:
                    topk = torch.topk(weights, k=3)
                    topk_indices = topk.indices.cpu().numpy()
                    topk_weights = topk.values.cpu().detach().numpy()
                    base_filename = "topk_slices_test.txt"
                    # Find next available id
                    id_num = 0
                    while True:
                        filename = f"{base_filename}.{id_num}" if id_num > 0 else base_filename
                        exists = False
                        if os.path.exists(filename):
                            # Check if this patient_id is already in the file
                            with open(filename, "r") as f:
                                for line in f:
                                    if line.startswith(f"{patient_id[0]}:"):
                                        exists = True
                                        break
                        if not exists:
                            break
                        id_num += 1
                    with open(filename, "a") as f:
                        f.write(f"{patient_id[0]}: indices={topk_indices.tolist()}, weights={topk_weights.tolist()}\n")

        acc = np.mean(np.array(preds) == np.array(labels))
        avg_val_loss = val_running_loss / len(test_loader)
        
        # Calculate validation metrics
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

        print(f"Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.4f} | Train Prec: {train_precision:.4f} | Train Rec: {train_recall:.4f} | Train F1: {train_f1:.4f} | Train AUC: {train_auc:.4f} | "
              f"Val Loss: {avg_val_loss:.4f} | Val Acc: {acc:.4f} | Val Prec: {precision:.4f} | Val Rec: {recall:.4f} | Val F1: {f1:.4f} | Val AUC: {auc:.4f}")

        # Save logs
        training_logs.append({
            "epoch": epoch + 1,
            "train_loss": avg_train_loss,
            "train_accuracy": train_acc,
            "train_precision": train_precision,
            "train_recall": train_recall,
            "train_f1": train_f1,
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

        # Save model after every epoch in 2d-aug directory
        epoch_ckpt_path = os.path.join(in_training_aug_dir, f"run_{run_id:03d}_epoch_{epoch+1}.pt")
        torch.save(model.state_dict(), epoch_ckpt_path)

        # Early stopping logic
        if f1 > best_val_f1:
            best_val_f1 = f1
            best_epoch = epoch
            epochs_no_improve = 0
            best_model_state = model.state_dict()
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= EARLY_STOPPING_PATIENCE:
            print(f"Early stopping at epoch {epoch+1}. Best val_f1: {best_val_f1:.4f} at epoch {best_epoch+1}")
            break
    
    # Save results for this run
    run_dir = os.path.join(output_dir, f"run_{run_id:03d}")
    os.makedirs(run_dir, exist_ok=True)
    
    # Save parameters
    with open(os.path.join(run_dir, "parameters.json"), "w") as f:
        json.dump(params, f, indent=4)
    
    # Save training logs
    with open(os.path.join(run_dir, "training_log.json"), "w") as f:
        json.dump(training_logs, f, indent=4)
    
    # Save best model
    if best_model_state is not None:
        torch.save(best_model_state, os.path.join(run_dir, "best_model.pt"))
    
    # Return best results
    return {
        'run_id': run_id,
        'parameters': params,
        'best_val_f1': best_val_f1,
        'best_epoch': best_epoch + 1,
        'final_val_auc': auc,
        'final_val_accuracy': acc,
        'final_val_precision': precision,
        'final_val_recall': recall,
        'training_logs': training_logs
    }

def run_single_training():
    """Run single training with default parameters"""
    print("=" * 60)
    print("RUNNING SINGLE TRAINING WITH DEFAULT PARAMETERS")
    print("=" * 60)
    print(f"Using weighted loss with pos_weight = {pos_weight:.4f} to handle class imbalance")
    print("=" * 60)
    
    # Default parameters
    default_params = {
        'weight_decay': 0.0001,
        'entropy_reg_weight': ENTROPY_REG_WEIGHT,
        'dropout_rate': 0.1,
        'attn_dropout_rate': 0.1
    }
    
    result = train_model_with_params(default_params, train_loader, test_loader, 0, OUTPUT_DIR)
    
    print(f"\nTraining completed!")
    print(f"Best F1 Score: {result['best_val_f1']:.4f}")
    print(f"Best Epoch: {result['best_epoch']}")
    print(f"Final Validation Metrics:")
    print(f"  Accuracy: {result['final_val_accuracy']:.4f}")
    print(f"  Precision: {result['final_val_precision']:.4f}")
    print(f"  Recall: {result['final_val_recall']:.4f}")
    print(f"  F1: {result['best_val_f1']:.4f}")
    print(f"  AUC: {result['final_val_auc']:.4f}")

# ---------- Main Execution ----------
def print_chunk_info():
    """Print information about all chunks"""
    param_names = list(PARAMETER_COMBINATIONS.keys())
    param_values = list(PARAMETER_COMBINATIONS.values())
    all_combinations = list(itertools.product(*param_values))
    total_combinations = len(all_combinations)
    total_chunks = (total_combinations + CHUNK_SIZE - 1) // CHUNK_SIZE
    
    print("=" * 60)
    print("PARAMETER SWEEP CHUNK INFORMATION")
    print("=" * 60)
    print(f"Total parameter combinations: {total_combinations}")
    print(f"Chunk size: {CHUNK_SIZE}")
    print(f"Total chunks: {total_chunks}")
    print(f"Current chunk: {CHUNK_INDEX + 1}")
    print()
    
    for chunk_idx in range(total_chunks):
        start_idx = chunk_idx * CHUNK_SIZE
        end_idx = min(start_idx + CHUNK_SIZE, total_combinations)
        print(f"Chunk {chunk_idx + 1}: combinations {start_idx + 1}-{end_idx}")
    print("=" * 60)

if __name__ == "__main__":
    run_single_training()

    # Debug: check model logits and attention weights for a single batch
    print("\n--- Debug: Single Batch Logits/Attention ---")
    model = CNNBackbone().to(DEVICE)
    model.train()
    for patient_slices, patient_label, patient_id in train_loader:
        patient_slices = patient_slices.squeeze(0).to(DEVICE)
        slice_logits, slice_attn = model(patient_slices)
        print("Slice logits:", slice_logits[:5].detach().cpu().numpy())
        print("Slice attn:", slice_attn[:5].detach().cpu().numpy())
        break