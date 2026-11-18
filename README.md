# Automated quality control for 3D breast MRI (MAMA-MIA)

Research code from a summer at the University of Chicago (Dr. Kyle Chard's group) on automated quality control of 3D breast tumor MRI from the MAMA-MIA dataset (1,506 patients). The main model is a 2D CNN with **attention-based slice aggregation**: each 3D volume is split into slices, a 2D backbone scores each slice, and a learned attention layer pools them into a volume-level prediction. Against a non-attention baseline it improved accuracy by 28%, precision by 36%, and AUC by 43%. The work contributed to a paper at eScience 2025.

## Pipeline

1. **Preprocess** (`preprocess/`): convert and resample volumes, crop to 128³ around the center of mass (`crop_and_save.py`), split train/test. SLURM job files included.
2. **Train** (`2d_cnn/main.py`): attention-pooled 2D CNN with early stopping, temperature scaling, entropy regularization on attention weights, and MONAI augmentations. K-fold CV with precision / recall / F1 / AUC reporting.
3. **Sweep** (`2d_cnn/parameter_sweep.py`, `focused_parameter_sweep.py`): regularization sweeps over weight decay, entropy weight, dropout, and attention dropout, chunked for cluster runs. See `2d_cnn/README_parameter_sweep.md`.
4. **Baselines** (`misc/`, `dense_mini/`, `res_mini/`, `cnn_mini/`): 3D DenseNet / ResNet / CNN baselines with their SLURM scripts.

## Run

```bash
cd 2d_cnn
pip install -r requirements.txt
python main.py                     # single training run
sbatch run_2d_cnn.slurm            # on a SLURM cluster (A100)
python analyze_sweep_results.py    # summarize a sweep
```

## Stack

PyTorch · MONAI · SimpleITK · scikit-learn · pandas · SLURM on NVIDIA A100 nodes
