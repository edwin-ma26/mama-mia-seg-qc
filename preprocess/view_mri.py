import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

def load_image_and_mask(file_path, image_channel=0, mask_channel=2):
    d = torch.load(file_path)
    tensor = d['tensor']  # shape [C, D, H, W]
    image = tensor[image_channel].numpy()
    mask = tensor[mask_channel].numpy()
    return image, mask


def show_slice(image, mask, slice_idx=None, axis=0, cmap='gray', alpha=0.4, save_path="output.png"):
    if slice_idx is None:
        slice_idx = image.shape[axis] // 2

    if axis == 0:
        img_slice = image[slice_idx, :, :]
        mask_slice = mask[slice_idx, :, :]
    elif axis == 1:
        img_slice = image[:, slice_idx, :]
        mask_slice = mask[:, slice_idx, :]
    elif axis == 2:
        img_slice = image[:, :, slice_idx]
        mask_slice = mask[:, :, slice_idx]
    else:
        raise ValueError("Axis must be 0, 1, or 2.")

    plt.figure(figsize=(6, 6))
    plt.imshow(img_slice, cmap=cmap)
    plt.imshow(mask_slice, cmap='Reds', alpha=alpha)
    plt.title(f"Slice {slice_idx} on axis {axis}")
    plt.axis('off')
    plt.savefig(save_path)
    print(f"Saved slice to {save_path}")
    plt.close()

if __name__ == "__main__":
    file_path = "preprocessed_train/DUKE_001_cropped.pt"
    image, mask = load_image_and_mask(file_path)
    show_slice(image, mask, axis=0, save_path="duke001_axial.png")
