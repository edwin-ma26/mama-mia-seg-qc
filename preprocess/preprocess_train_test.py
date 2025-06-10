import os
import torch
import numpy as np
import nibabel as nib
import SimpleITK as sitk
import pandas as pd
from scipy.ndimage import zoom
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context

# ---------- Configuration ----------
image_dir = "../images"
mask_dir = "../segmentations/automatic"
split_csv = "../train_test_splits.csv"
label_csv = "../preliminary_automatic_segmentations_quality_scores.csv"

# ---------- Load Labels (Top-level, for multiprocessing) ----------
label_df = pd.read_csv(label_csv)
label_map = {
    row['patient_id']: 1 if row['expert_1_qs'] == 'Good' and row['expert_2_qs'] == 'Good' else 0
    for _, row in label_df.iterrows()
}

target_shape = (128, 128, 128)

save_dirs = {
    "train": "preprocessed_train",
    "test": "preprocessed_test"
}

# ---------- Preprocessing Functions ----------
def load_nifti(path):
    img = nib.load(path)
    return img.get_fdata(), img.header.get_zooms()

def n4_bias_correction(image_np):
    sitk_image = sitk.GetImageFromArray(image_np.astype(np.float32))
    mask_image = sitk.OtsuThreshold(sitk_image, 0, 1, 200)
    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    corrected = corrector.Execute(sitk_image, mask_image)
    return sitk.GetArrayFromImage(corrected)

def denoise_image(image_np):
    sitk_image = sitk.GetImageFromArray(image_np.astype(np.float32))
    denoised = sitk.CurvatureFlow(image1=sitk_image, timeStep=0.125, numberOfIterations=5)
    return sitk.GetArrayFromImage(denoised)

def resample_image(img_np, original_spacing, new_spacing=[1.0, 1.0, 1.0]):
    resize_factor = np.array(original_spacing) / np.array(new_spacing)
    new_shape = np.round(np.array(img_np.shape) * resize_factor).astype(int)
    zoom_factors = new_shape / np.array(img_np.shape)
    return zoom(img_np, zoom_factors, order=1)

def is_already_processed(pid, split):
    return os.path.exists(os.path.join(save_dirs[split], f"{pid}_resized.pt")) and \
           os.path.exists(os.path.join(save_dirs[split], f"{pid}_resampled.pt"))

def preprocess_and_save(pid, split):
    try:
        img_path_0 = os.path.join(image_dir, pid, f"{pid}_0000.nii.gz")
        img_path_1 = os.path.join(image_dir, pid, f"{pid}_0001.nii.gz")
        mask_path = os.path.join(mask_dir, f"{pid}.nii.gz")
        resized_save_path = os.path.join(save_dirs[split], f"{pid}_resized.pt")
        resampled_save_path = os.path.join(save_dirs[split], f"{pid}_resampled.pt")

        for path in [img_path_0, img_path_1, mask_path]:
            if not os.path.exists(path):
                print(f"[SKIP][{split}] {pid}: Missing file {path}")
                return

        label = label_map.get(pid)
        if label is None:
            print(f"[SKIP][{split}] {pid}: Missing label.")
            return

        print(f"[{split.upper()}] Processing {pid}...", flush=True)

        mri0_np, spacing = load_nifti(img_path_0)
        mri1_np, _ = load_nifti(img_path_1)
        mask_np, _ = load_nifti(mask_path)

        mri0_np = denoise_image(n4_bias_correction(mri0_np))
        mri1_np = denoise_image(n4_bias_correction(mri1_np))

        mri0_np = resample_image(mri0_np, spacing)
        mri1_np = resample_image(mri1_np, spacing)
        mask_np = resample_image(mask_np, spacing)

        stacked_resampled = np.stack([mri0_np, mri1_np, (mask_np > 0).astype(np.float32)], axis=0)
        tensor_resampled = torch.tensor(stacked_resampled, dtype=torch.float32)
        torch.save({"tensor": tensor_resampled, "label": torch.tensor([label], dtype=torch.float32)}, resampled_save_path)

        mri0_np = zoom(mri0_np, np.array(target_shape) / np.array(mri0_np.shape), order=1)
        mri1_np = zoom(mri1_np, np.array(target_shape) / np.array(mri1_np.shape), order=1)
        mask_np = zoom(mask_np, np.array(target_shape) / np.array(mask_np.shape), order=0)
        mask_np = (mask_np > 0).astype(np.float32)

        stacked_resized = np.stack([mri0_np, mri1_np, mask_np], axis=0)
        tensor_resized = torch.tensor(stacked_resized, dtype=torch.float32)
        torch.save({"tensor": tensor_resized, "label": torch.tensor([label], dtype=torch.float32)}, resized_save_path)

        print(f"[{split.upper()}] ✅ Saved: {resampled_save_path}, {resized_save_path}", flush=True)

    except Exception as e:
        print(f"[ERROR][{split}] {pid}: {e}", flush=True)

# ---------- Run Parallel ----------
def run_parallel(split_name, pid_list, max_to_process=None):
    print(f"[MULTI] Starting parallel preprocessing for {split_name} with {len(pid_list)} patients...", flush=True)
    filtered_pids = [pid for pid in pid_list if not is_already_processed(pid, split_name)]

    if max_to_process:
        filtered_pids = filtered_pids[:max_to_process]
        print(f"[INFO] Limiting to {max_to_process} unprocessed samples.")

    ctx = get_context("spawn")  # safer across platforms and SLURM
    with ProcessPoolExecutor(max_workers=8, mp_context=ctx) as executor:
        futures = [executor.submit(preprocess_and_save, pid, split_name) for pid in filtered_pids]
        for f in futures:
            try:
                f.result()
            except Exception as e:
                print(f"[MULTI][ERROR] {e}", flush=True)

# ---------- Main Execution ----------
if __name__ == "__main__":
    os.makedirs(save_dirs["train"], exist_ok=True)
    os.makedirs(save_dirs["test"], exist_ok=True)

    split_df = pd.read_csv(split_csv)
    train_ids = split_df['train_split'].dropna().values
    test_ids = split_df['test_split'].dropna().values

    run_parallel("train", train_ids, max_to_process=100)
    run_parallel("test", test_ids, max_to_process=20)

    print("✅ All preprocessing complete.")
