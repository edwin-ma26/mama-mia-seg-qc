#!/bin/bash
#SBATCH --job-name=mri-3d-cnn
#SBATCH --output=cnn_output.txt
#SBATCH --error=cnn_error.txt
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --account=cis230030-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00

# Activate conda environment
source ~/.bashrc
conda activate cnn-env

# Go to your script directory
cd $SLURM_SUBMIT_DIR

# Run the script
python 3d_cnn.py
