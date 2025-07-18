import os
import numpy as np
import pandas as pd
import SimpleITK as sitk
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from sklearn.model_selection import KFold
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from tqdm import tqdm
import json
import itertools
from datetime import datetime

# ---------- Config ----------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 25  # Reduced for focused sweep
IMG_SIZE = (384, 384)
OUTPUT_DIR = "./focused_sweep_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Early stopping config
EARLY_STOPPING_PATIENCE = 5

# Temperature scaling config
TEMPERATURE = 1.0

# ---------- Focused Parameter Sweep Config ----------
# More targeted parameter ranges based on common best practices
FOCUSED_PARAMETER_COMBINATIONS = {
    'weight_decay': [0.0, 0.0001, 0.001],  # L2 regularization - focus on smaller values
    'entropy_reg_weight': [0.0, 0.0005, 0.001],  # Entropy regularization - focus on moderate values
    'dropout_rate': [0.1, 0.2, 0.3],  # Dropout rate - focus on common effective values
    'attn_dropout_rate': [0.0, 0.1, 0.2],  # Attention dropout - focus on smaller values
}

# ---------- Model ----------
class ResNetBackbone(nn.Module):
    def __init__(self, dropout_rate=0.1, attn_dropout=0.1):
        super(ResNetBackbone, self).__init__()
        self.base = models.resnet18(pretrained=True)
        self.base.conv1 = nn.Conv2d(2, 64, kernel_size=7, stride=2, padding=3, bias=False)

        num_feats = self.base.fc.in_features
        self.base.fc = nn.Identity()

        self.dropout = nn.Dropout(dropout_rate)
        self.slice_classifier = nn.Linear(num_feats, 1)
        self.slice_attn = nn.Linear(num_feats, 1)
        self.attn_dropout = nn.Dropout(attn_dropout)

    def forward(self, x):
        x = self.base(x)  # [batch, features]
        x = self.dropout(x)
        logit = self.slice_classifier(x).squeeze(-1)
        attn_weight = self.slice_attn(x).squeeze(-1)
        attn_weight = self.attn_dropout(attn_weight)
        return logit, attn_weight

# ---------- Dataset ----------
class AllSlicesDataset(Dataset):
    def __init__(self, patient_ids, img_size=IMG_SIZE):
        self.samples = []
        qs_df = pd.read_csv("../preliminary_automatic_segmentations_quality_scores.csv", dtype=str)
        qs_df = qs_df.set_index("patient_id")

        # Track statistics
        original_count = 0
        augmented_count = 0
        patients_processed = 0

        for pid in patient_ids:
            if pid not in qs_df.index:
                continue

            expert_1 = qs_df.loc[pid, "expert_1_qs"]
            expert_2 = qs_df.loc[pid, "expert_2_qs"]
            label = 0 if (expert_1 == "Good" and expert_2 == "Good") else 1

            # List of data sources to try (original + augmented)
            data_sources = []
            
            # Original data
            img_path = f"../preprocessed/{pid}/{pid}_n4_denoised_resampled_padded.nii.gz"
            mask_path = f"../preprocessed/{pid}/{pid}_mask_resampled_padded.nii.gz"
            if os.path.exists(img_path) and os.path.exists(mask_path):
                data_sources.append((img_path, mask_path, "original"))
            
            # Augmented data (aug_07, aug_08, aug_09)
            for aug_num in ["07", "08", "09"]:
                aug_img_path = f"../preprocessed/{pid}/{pid}_n4_denoised_resampled_padded_aug_{aug_num}.nii.gz"
                aug_mask_path = f"../preprocessed/{pid}/{pid}_mask_resampled_padded_aug_{aug_num}.nii.gz"
                if os.path.exists(aug_img_path) and os.path.exists(aug_mask_path):
                    data_sources.append((aug_img_path, aug_mask_path, f"aug_{aug_num}"))

            # Process all available data sources
            for img_path, mask_path, source_type in data_sources:
                img = sitk.GetArrayFromImage(sitk.ReadImage(img_path))
                mask = sitk.GetArrayFromImage(sitk.ReadImage(mask_path))

                slices = []
                # Process all slices in the volume
                for i in range(mask.shape[0]):
                    if mask[i, :, :].sum() == 0:
                        continue
                    img_slice = img[i, :, :]
                    mask_slice = mask[i, :, :]
                    slice_2ch = np.stack([img_slice, mask_slice], axis=0)
                    slice_tensor = torch.tensor(slice_2ch, dtype=torch.float32)
                    slices.append(slice_tensor)

                if len(slices) > 0:
                    patient_tensor = torch.stack(slices)  # [Num_slices, 2, H, W]
                    self.samples.append((patient_tensor, torch.tensor(label, dtype=torch.float32)))
                    
                    # Track statistics
                    if source_type == "original":
                        original_count += 1
                    else:
                        augmented_count += 1

            if len(data_sources) > 0:
                patients_processed += 1

        print(f"Dataset statistics: {original_count} original samples, {augmented_count} augmented samples from {patients_processed} patients")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

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

def train_model_with_params(params, train_loader, test_loader, run_id):
    """Train a model with specific regularization parameters"""
    
    # Create model with specified dropout rates
    model = ResNetBackbone(
        dropout_rate=params['dropout_rate'],
        attn_dropout=params['attn_dropout_rate']
    ).to(DEVICE)
    
    # Create optimizer with specified weight decay
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=1e-4, 
        weight_decay=params['weight_decay']
    )
    criterion = nn.BCEWithLogitsLoss()
    
    # Training logs
    training_logs = []
    best_val_f1 = -1
    best_epoch = -1
    epochs_no_improve = 0
    best_model_state = None
    
    print(f"\n{'='*60}")
    print(f"RUN {run_id}: weight_decay={params['weight_decay']}, "
          f"entropy_reg={params['entropy_reg_weight']}, "
          f"dropout={params['dropout_rate']}, "
          f"attn_dropout={params['attn_dropout_rate']}")
    print(f"{'='*60}")
    
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        train_preds, train_labels = [], []
        train_logits = []
        
        for patient_slices, patient_label in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
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
            
            # Total loss
            total_loss = classification_loss + entropy_loss
            
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
            for patient_slices, patient_label in test_loader:
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

        print(f"Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.4f} | "
              f"Val Acc: {acc:.4f} | Val F1: {f1:.4f} | Val AUC: {auc:.4f}")

        # Save logs
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
    
    # Return best results
    return {
        'best_val_f1': best_val_f1,
        'best_epoch': best_epoch + 1,
        'final_val_auc': auc,
        'final_val_accuracy': acc,
        'final_val_precision': precision,
        'final_val_recall': recall,
        'training_logs': training_logs,
        'best_model_state': best_model_state
    }

def main():
    # Load datasets
    train_file = '../filter/train_patients.txt'
    test_file = '../filter/test_patients.txt'

    train_patient_ids = load_patient_ids(train_file)
    test_patient_ids = load_patient_ids(test_file)

    print(f"Loading training dataset with {len(train_patient_ids)} patients...")
    train_dataset = AllSlicesDataset(train_patient_ids)
    print(f"Training dataset loaded: {len(train_dataset)} samples")

    print(f"Loading test dataset with {len(test_patient_ids)} patients...")
    test_dataset = AllSlicesDataset(test_patient_ids)
    print(f"Test dataset loaded: {len(test_dataset)} samples")

    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # Generate all parameter combinations
    param_names = list(FOCUSED_PARAMETER_COMBINATIONS.keys())
    param_values = list(FOCUSED_PARAMETER_COMBINATIONS.values())
    all_combinations = list(itertools.product(*param_values))
    
    print(f"\nTotal parameter combinations to test: {len(all_combinations)}")
    print("Parameter combinations:")
    for i, combo in enumerate(all_combinations):
        params = dict(zip(param_names, combo))
        print(f"  {i+1}. WD: {params['weight_decay']}, ER: {params['entropy_reg_weight']}, "
              f"DR: {params['dropout_rate']}, ADR: {params['attn_dropout_rate']}")
    
    # Results storage
    all_results = []
    
    # Create timestamp for this sweep
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    sweep_dir = os.path.join(OUTPUT_DIR, f"focused_sweep_{timestamp}")
    os.makedirs(sweep_dir, exist_ok=True)
    
    # Run parameter sweep
    for run_id, param_combo in enumerate(all_combinations):
        params = dict(zip(param_names, param_combo))
        
        # Create run directory
        run_dir = os.path.join(sweep_dir, f"run_{run_id:03d}")
        os.makedirs(run_dir, exist_ok=True)
        
        # Save parameters
        with open(os.path.join(run_dir, "parameters.json"), "w") as f:
            json.dump(params, f, indent=4)
        
        try:
            # Train model with these parameters
            results = train_model_with_params(params, train_loader, test_loader, run_id)
            
            # Save results
            results['run_id'] = run_id
            results['parameters'] = params
            
            # Save training logs
            with open(os.path.join(run_dir, "training_log.json"), "w") as f:
                json.dump(results['training_logs'], f, indent=4)
            
            # Save best model
            if results['best_model_state'] is not None:
                torch.save(results['best_model_state'], os.path.join(run_dir, "best_model.pt"))
            
            all_results.append(results)
            
            print(f"Run {run_id} completed. Best F1: {results['best_val_f1']:.4f}")
            
        except Exception as e:
            print(f"Error in run {run_id}: {str(e)}")
            all_results.append({
                'run_id': run_id,
                'parameters': params,
                'error': str(e),
                'best_val_f1': -1
            })
    
    # Create summary
    summary_data = []
    for result in all_results:
        if 'error' not in result:
            summary_data.append({
                'run_id': result['run_id'],
                'weight_decay': result['parameters']['weight_decay'],
                'entropy_reg_weight': result['parameters']['entropy_reg_weight'],
                'dropout_rate': result['parameters']['dropout_rate'],
                'attn_dropout_rate': result['parameters']['attn_dropout_rate'],
                'best_val_f1': result['best_val_f1'],
                'best_epoch': result['best_epoch'],
                'final_val_auc': result['final_val_auc'],
                'final_val_accuracy': result['final_val_accuracy'],
                'final_val_precision': result['final_val_precision'],
                'final_val_recall': result['final_val_recall']
            })
        else:
            summary_data.append({
                'run_id': result['run_id'],
                'weight_decay': result['parameters']['weight_decay'],
                'entropy_reg_weight': result['parameters']['entropy_reg_weight'],
                'dropout_rate': result['parameters']['dropout_rate'],
                'attn_dropout_rate': result['parameters']['attn_dropout_rate'],
                'best_val_f1': -1,
                'best_epoch': -1,
                'final_val_auc': -1,
                'final_val_accuracy': -1,
                'final_val_precision': -1,
                'final_val_recall': -1,
                'error': result['error']
            })
    
    # Save summary
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(os.path.join(sweep_dir, "focused_parameter_sweep_summary.csv"), index=False)
    
    # Find best parameters
    valid_results = [r for r in all_results if 'error' not in r]
    if valid_results:
        best_result = max(valid_results, key=lambda x: x['best_val_f1'])
        print(f"\n{'='*60}")
        print("BEST PARAMETERS FOUND:")
        print(f"{'='*60}")
        print(f"Best F1 Score: {best_result['best_val_f1']:.4f}")
        print(f"Best Parameters:")
        for param, value in best_result['parameters'].items():
            print(f"  {param}: {value}")
        print(f"Best Epoch: {best_result['best_epoch']}")
        print(f"Final AUC: {best_result['final_val_auc']:.4f}")
        print(f"Final Accuracy: {best_result['final_val_accuracy']:.4f}")
        
        # Save best parameters
        with open(os.path.join(sweep_dir, "best_parameters.json"), "w") as f:
            json.dump({
                'best_f1': best_result['best_val_f1'],
                'best_epoch': best_result['best_epoch'],
                'final_auc': best_result['final_val_auc'],
                'final_accuracy': best_result['final_val_accuracy'],
                'parameters': best_result['parameters']
            }, f, indent=4)
    
    # Print all results sorted by F1 score
    print(f"\n{'='*60}")
    print("ALL RESULTS (sorted by F1 score):")
    print(f"{'='*60}")
    sorted_results = sorted(valid_results, key=lambda x: x['best_val_f1'], reverse=True)
    for i, result in enumerate(sorted_results):
        print(f"{i+1}. F1: {result['best_val_f1']:.4f} | "
              f"WD: {result['parameters']['weight_decay']} | "
              f"ER: {result['parameters']['entropy_reg_weight']} | "
              f"DR: {result['parameters']['dropout_rate']} | "
              f"ADR: {result['parameters']['attn_dropout_rate']} | "
              f"AUC: {result['final_val_auc']:.4f}")
    
    print(f"\nFocused parameter sweep completed. Results saved to: {sweep_dir}")

if __name__ == "__main__":
    main() 