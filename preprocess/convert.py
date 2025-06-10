import torch
import nibabel as nib
import numpy as np
import os

def convert_pt_to_nii(pt_path, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    
    data = torch.load(pt_path)  # dict with keys ['tensor', 'label']
    tensor = data['tensor']     # shape [C, D, H, W]

    for ch in range(tensor.shape[0]):
        volume = tensor[ch].numpy()  # shape [D, H, W]
        
        # Create a NIfTI image
        nii_img = nib.Nifti1Image(volume, affine=np.eye(4))
        
        # Save to file
        out_path = os.path.join(out_dir, f"{os.path.basename(pt_path).replace('.pt', '')}_ch{ch}.nii.gz")
        nib.save(nii_img, out_path)
        print(f"Saved channel {ch} to {out_path}")

# Example usage
if __name__ == "__main__":
    pt_file = "preprocessed_train/DUKE_001_cropped.pt"
    output_dir = "nifti_output"
    convert_pt_to_nii(pt_file, output_dir)
