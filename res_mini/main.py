import os
import time
import psutil
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import sys
sys.path.append("../3D-CNN-PyTorch")
from models import resnet

# ---------- Dataset ----------
class MRIDataset(Dataset):
    def __init__(self, data_dir):
        print(f"[INFO] Initializing MRIDataset for: {data_dir}", flush=True)
        # Only use files ending in _resampled.pt
        self.paths = sorted([
            os.path.join(data_dir, f)
            for f in os.listdir(data_dir)
            if f.endswith("cropped.pt")
        ])
        print(f"[INFO] Found {len(self.paths)} cropped.pt files in {data_dir}", flush=True)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        if idx % 10 == 0:
            print(f"[DATA] Loading .pt file #{idx}: {self.paths[idx]}", flush=True)
        data = torch.load(self.paths[idx])
        tensor = data["tensor"][[0, 2], ...]  # Select channels 0 and 2 only
        label = data["label"]
        return tensor, label


# ---------- Model ----------
def generate_model():
    print("[MODEL] Initializing ResNet model...", flush=True)
    model = resnet.generate_model(
        model_depth=50,
        n_input_channels=2,
        n_classes=1,
        conv1_t_size=7,
        conv1_t_stride=1,
        no_max_pool=False,
        shortcut_type='B',
        widen_factor=1.0
    )
    print("[MODEL] Model initialized.", flush=True)
    return model

# ---------- Training ----------
def train_single_gpu():
    print("[MAIN] Starting training script...", flush=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[MAIN] Using device: {device}", flush=True)

    print("[DATA] Loading dataset...", flush=True)
    full_dataset = MRIDataset("../preprocess/preprocessed_train")

    print("[DATA] Splitting into train/val sets...", flush=True)
    val_size = int(0.2 * len(full_dataset))
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    print(f"[DATA] Train size: {train_size}, Val size: {val_size}", flush=True)

    print("[DATA] Initializing dataloaders...", flush=True)
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=0, pin_memory=True)
    print("[DATA] Dataloaders ready.", flush=True)

    print("[MODEL] Generating model...", flush=True)
    model = generate_model().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.BCEWithLogitsLoss()

    torch.backends.cudnn.benchmark = True
    start_time = time.time()

    num_epochs = 10  # ✅ Fixed indentation

    print("[TRAIN] Starting training loop...", flush=True)
    for epoch in range(num_epochs):
        print(f"\n[TRAIN] === Epoch {epoch+1}/{num_epochs} ===", flush=True)
        torch.cuda.reset_peak_memory_stats()
        model.train()

        for i, (inputs, labels) in enumerate(train_loader):
            batch_start = time.time()
            print(f"[TRAIN] --- Batch {i} ---", flush=True)
            inputs, labels = inputs.to(device), labels.to(device).view(-1)

            outputs = model(inputs).view(-1)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            try:
                loss.backward()
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print("[ERROR] CUDA out of memory during backward pass.", flush=True)
                    exit(1)
                else:
                    raise

            optimizer.step()

            preds = torch.sigmoid(outputs).detach().cpu().numpy()
            targets = labels.detach().cpu().numpy()
            print(f"[TRAIN][Epoch {epoch+1}][Batch {i:03d}] Loss: {loss.item():.4f} | Preds: {np.round(preds, 2)} | Targets: {targets}", flush=True)
            print(f"[TRAIN] Batch {i} complete. Time: {time.time() - batch_start:.2f}s\n", flush=True)

        # ---------- Validation ----------
        print(f"[VAL] Starting validation for epoch {epoch+1}...", flush=True)
        model.eval()
        val_losses = []
        val_correct = 0
        with torch.no_grad():
            for i, (inputs, labels) in enumerate(val_loader):
                if i % 10 == 0:
                    print(f"[VAL] Batch {i}", flush=True)
                inputs, labels = inputs.to(device), labels.to(device).view(-1)
                outputs = model(inputs).view(-1)
                loss = criterion(outputs, labels)
                val_losses.append(loss.item())

                preds = (torch.sigmoid(outputs) > 0.5).long()
                val_correct += (preds == labels.long()).sum().item()

        val_loss = np.mean(val_losses)
        val_acc = val_correct / len(val_dataset)
        print(f"[VAL][Epoch {epoch+1}] Loss: {val_loss:.4f} | Accuracy: {val_acc:.4f}", flush=True)

        # GPU Memory Reporting
        peak_mem = torch.cuda.max_memory_allocated() / 1e6  # MB
        print(f"[RESOURCE][Epoch {epoch+1}] GPU Peak Memory: {peak_mem:.2f} MB", flush=True)

        # Save checkpoint
        os.makedirs("checkpoints", exist_ok=True)
        checkpoint_path = f"checkpoints/resnet_single_gpu_epoch{epoch+1}.pth"
        torch.save(model.state_dict(), checkpoint_path)
        print(f"[SAVE] Model checkpoint saved to '{checkpoint_path}'", flush=True)

    total_time = time.time() - start_time
    print(f"[TIME] Total training time: {total_time:.2f} seconds", flush=True)


# ---------- Entry Point ----------
if __name__ == "__main__":
    print("[MAIN] Calling train_single_gpu()...", flush=True)
    train_single_gpu()
