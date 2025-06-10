import os
import torch
import numpy as np
from scipy.ndimage import center_of_mass
import torch.nn.functional as F

# ---------- Config ----------
TARGET_SHAPE = (128, 128, 128)
INPUT_SUFFIX = "_resampled.pt"
OUTPUT_SUFFIX = "_cropped.pt"
DATA_FOLDERS = ["preprocessed_train", "preprocessed_test"]

def crop_with_padding(vol, center, target_shape):
    dz, dy, dx = target_shape
    cz, cy, cx = center
    zmin, ymin, xmin = cz - dz // 2, cy - dy // 2, cx - dx // 2
    zmax, ymax, xmax = zmin + dz, ymin + dy, xmin + dx

    # Compute required padding for each axis
    pad = [
        (max(0, -zmin), max(0, zmax - vol.shape[1])),
        (max(0, -ymin), max(0, ymax - vol.shape[2])),
        (max(0, -xmin), max(0, xmax - vol.shape[3]))
    ]

    if any(p[0] > 0 or p[1] > 0 for p in pad):
        vol = F.pad(vol, (pad[2][0], pad[2][1], pad[1][0], pad[1][1], pad[0][0], pad[0][1]), mode='reflect')

    # Adjust crop region after padding
    zmin = max(zmin, 0)
    ymin = max(ymin, 0)
    xmin = max(xmin, 0)

    return vol[:, zmin:zmin+dz, ymin:ymin+dy, xmin:xmin+dx]

def crop_and_save_all(folder_path):
    pt_files = [f for f in os.listdir(folder_path) if f.endswith(INPUT_SUFFIX)]

    for fname in pt_files:
        fpath = os.path.join(folder_path, fname)
        try:
            data = torch.load(fpath)
            tensor = data["tensor"]  # Shape: [3, D, H, W]
            label = data["label"]

            mask_np = tensor[2].numpy()
            if mask_np.sum() == 0:
                center = [s // 2 for s in mask_np.shape]
            else:
                center = np.round(center_of_mass(mask_np)).astype(int)

            cropped_tensor = crop_with_padding(tensor, center, TARGET_SHAPE)
            save_path = os.path.join(folder_path, fname.replace(INPUT_SUFFIX, OUTPUT_SUFFIX))
            torch.save({"tensor": cropped_tensor, "label": label}, save_path)
            print(f"[✔] Saved cropped tensor: {save_path}")

        except Exception as e:
            print(f"[✘] Error processing {fname}: {e}")

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    for folder in DATA_FOLDERS:
        print(f"\n[INFO] Processing folder: {folder}")
        crop_and_save_all(os.path.join(base_dir, folder))
