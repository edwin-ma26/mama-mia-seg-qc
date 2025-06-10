import re
import matplotlib.pyplot as plt
import os

# File path and output directory
log_path = "res_mini/logs/output_11626989.txt"
output_dir = "res_mini/logs"

# Read log content
with open(log_path, "r") as f:
    log_text = f.read()

# Extract training losses per batch
pattern = r"\[TRAIN\]\[Epoch (\d+)\]\[Batch \d+\] Loss: ([\d\.]+)"
matches = re.findall(pattern, log_text)

# Organize by epoch
epoch_losses = {}
for epoch, loss in matches:
    epoch = int(epoch)
    loss = float(loss)
    epoch_losses.setdefault(epoch, []).append(loss)

# Create one figure with subplots
num_epochs = len(epoch_losses)
fig, axs = plt.subplots(num_epochs, 1, figsize=(10, 3 * num_epochs), sharex=False)

for idx, epoch in enumerate(sorted(epoch_losses)):
    ax = axs[idx] if num_epochs > 1 else axs
    ax.plot(epoch_losses[epoch], marker='o')
    ax.set_title(f"Epoch {epoch}")
    ax.set_xlabel("Batch")
    ax.set_ylabel("Loss")
    ax.grid(True)

plt.tight_layout()
combined_path = os.path.join(output_dir, "combined_loss_plot.png")
plt.savefig(combined_path)
plt.close()

print(f"Saved combined loss plot to: {combined_path}")
