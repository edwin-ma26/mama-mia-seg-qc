import re
import matplotlib.pyplot as plt
from collections import defaultdict
import os

# Log filename
log_filename = "output_11748977.txt"

# Load log text
with open(log_filename, "r") as f:
    log_text = f.read()

# Extract training losses
train_pattern = r"\[TRAIN\]\[Epoch (\d+)\]\[Batch \d+\] Loss: ([\d\.]+)"
train_matches = re.findall(train_pattern, log_text)

train_losses_by_epoch = defaultdict(list)
for epoch, loss in train_matches:
    train_losses_by_epoch[int(epoch)].append(float(loss))

# Extract validation losses
val_pattern = r"\[VAL\]\[Epoch (\d+)\] Loss: ([\d\.]+)"
val_matches = re.findall(val_pattern, log_text)
val_losses_by_epoch = {int(epoch): float(loss) for epoch, loss in val_matches}

# Compute average training loss per epoch
epochs = sorted(train_losses_by_epoch.keys())
avg_train_losses = [sum(train_losses_by_epoch[epoch]) / len(train_losses_by_epoch[epoch]) for epoch in epochs]
val_losses = [val_losses_by_epoch.get(epoch, None) for epoch in epochs]

# Plot
plt.figure(figsize=(10, 6))
plt.plot(epochs, avg_train_losses, label='Train Loss', marker='o')
plt.plot(epochs, val_losses, label='Val Loss', marker='s')
plt.title("Average Training and Validation Loss Per Epoch")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)
plt.legend()
plt.xticks(epochs)
plt.tight_layout()

# Save with filename suffix
base_filename = os.path.splitext(os.path.basename(log_filename))[0]
save_path = f"loss_per_epoch_{base_filename}.png"
plt.savefig(save_path)
plt.show()
