import os
import time
import psutil
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
from collections import OrderedDict

# ---------- Dataset ----------
class MRIDataset(Dataset):
    def __init__(self, data_dir):
        print(f"[INFO] Initializing MRIDataset for: {data_dir}", flush=True)
        self.paths = sorted([os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(".pt")])
        print(f"[INFO] Found {len(self.paths)} .pt files in {data_dir}", flush=True)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        if idx % 10 == 0:
            print(f"[DATA] Loading .pt file #{idx}: {self.paths[idx]}", flush=True)
        data = torch.load(self.paths[idx])
        return data["tensor"], data["label"]

# ---------- DenseNet ----------
class _DenseLayer(nn.Sequential):
    def __init__(self, in_features, growth_rate, bn_size, drop_rate):
        super().__init__()
        self.add_module('norm1', nn.BatchNorm3d(in_features))
        self.add_module('relu1', nn.ReLU(inplace=True))
        self.add_module('conv1', nn.Conv3d(in_features, bn_size * growth_rate, 1, bias=False))
        self.add_module('norm2', nn.BatchNorm3d(bn_size * growth_rate))
        self.add_module('relu2', nn.ReLU(inplace=True))
        self.add_module('conv2', nn.Conv3d(bn_size * growth_rate, growth_rate, 3, padding=1, bias=False))
        self.drop_rate = drop_rate

    def forward(self, x):
        new_features = super().forward(x)
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return torch.cat([x, new_features], 1)

class _DenseBlock(nn.Sequential):
    def __init__(self, num_layers, in_features, bn_size, growth_rate, drop_rate):
        super().__init__()
        for i in range(num_layers):
            layer = _DenseLayer(in_features + i * growth_rate, growth_rate, bn_size, drop_rate)
            self.add_module(f'denselayer{i+1}', layer)

class _Transition(nn.Sequential):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.add_module('norm', nn.BatchNorm3d(in_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv3d(in_features, out_features, 1, bias=False))
        self.add_module('pool', nn.AvgPool3d(2, stride=2))

class DenseNet(nn.Module):
    def __init__(self, in_channels=2, num_classes=1, growth_rate=32, block_config=(6,12,24,16),
                 init_features=64, bn_size=4, drop_rate=0):
        super().__init__()
        self.features = [('conv1', nn.Conv3d(in_channels, init_features, kernel_size=(7,7,7), stride=2, padding=3, bias=False)),
                         ('norm1', nn.BatchNorm3d(init_features)),
                         ('relu1', nn.ReLU(inplace=True)),
                         ('pool1', nn.MaxPool3d(3, stride=2, padding=1))]
        self.features = nn.Sequential(OrderedDict(self.features))

        num_features = init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(num_layers, num_features, bn_size, growth_rate, drop_rate)
            self.features.add_module(f'denseblock{i+1}', block)
            num_features += num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition(num_features, num_features // 2)
                self.features.add_module(f'transition{i+1}', trans)
                num_features = num_features // 2

        self.features.add_module('norm_final', nn.BatchNorm3d(num_features))
        self.classifier = nn.Linear(num_features, num_classes)

    def forward(self, x):
        features = self.features(x)
        out = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool3d(out, 1).view(x.size(0), -1)
        return self.classifier(out)

# ---------- Model Wrapper ----------
def generate_model():
    print("[MODEL] Initializing DenseNet model...", flush=True)
    model = DenseNet()
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

    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)
    print("[DATA] Dataloaders ready.", flush=True)

    model = generate_model().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.BCEWithLogitsLoss()

    start_time = time.time()

    num_epochs = 5
    for epoch in range(num_epochs):
        print(f"\n[TRAIN] === Epoch {epoch+1}/{num_epochs} ===", flush=True)
        torch.cuda.reset_peak_memory_stats()
        model.train()

        for i, (inputs, labels) in enumerate(train_loader):
            print(f"[TRAIN] --- Batch {i} ---", flush=True)
            inputs, labels = inputs.to(device), labels.to(device).view(-1).float()

            optimizer.zero_grad()
            outputs = model(inputs).view(-1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            preds = torch.sigmoid(outputs).detach().cpu().numpy()
            targets = labels.detach().cpu().numpy()
            print(f"[TRAIN][Epoch {epoch+1}][Batch {i:03d}] Loss: {loss.item():.4f} | Preds: {np.round(preds, 2)} | Targets: {targets}", flush=True)

        # ---------- Validation ----------
        print(f"[VAL] Starting validation for epoch {epoch+1}...", flush=True)
        model.eval()
        val_losses = []
        val_correct = 0
        with torch.no_grad():
            for i, (inputs, labels) in enumerate(val_loader):
                if i % 10 == 0:
                    print(f"[VAL] Batch {i}", flush=True)
                inputs, labels = inputs.to(device), labels.to(device).view(-1).float()
                outputs = model(inputs).view(-1)
                loss = criterion(outputs, labels)
                val_losses.append(loss.item())

                preds = (torch.sigmoid(outputs) > 0.5).long()
                val_correct += (preds == labels.long()).sum().item()

        val_loss = np.mean(val_losses)
        val_acc = val_correct / len(val_dataset)
        print(f"[VAL][Epoch {epoch+1}] Loss: {val_loss:.4f} | Accuracy: {val_acc:.4f}", flush=True)

        peak_mem = torch.cuda.max_memory_allocated() / 1e6
        print(f"[RESOURCE][Epoch {epoch+1}] GPU Peak Memory: {peak_mem:.2f} MB", flush=True)

        os.makedirs("checkpoints", exist_ok=True)
        checkpoint_path = f"checkpoints/densenet_single_gpu_epoch{epoch+1}.pth"
        torch.save(model.state_dict(), checkpoint_path)
        print(f"[SAVE] Model checkpoint saved to '{checkpoint_path}'", flush=True)

    total_time = time.time() - start_time
    print(f"[TIME] Total training time: {total_time:.2f} seconds", flush=True)

# ---------- Entry ----------
if __name__ == "__main__":
    print("[MAIN] Calling train_single_gpu()...", flush=True)
    train_single_gpu()
