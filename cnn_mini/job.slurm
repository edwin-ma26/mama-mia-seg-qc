#!/bin/bash
#SBATCH --job-name=single_gpu_train
#SBATCH --output=logs/output_%j.txt
#SBATCH --error=logs/error_%j.txt
#SBATCH --partition=gpu
#SBATCH --account=cis230030-gpu        # Replace with your actual GPU account if needed
#SBATCH --gres=gpu:1                   # Request 1 GPU
#SBATCH --cpus-per-task=8              # Adjust based on num_workers in DataLoader
#SBATCH --mem=64G                      # Adjust memory as needed
#SBATCH --time=05:00:00                # Adjust time limit as needed
#SBATCH --nodes=1
#SBATCH --ntasks=1

# Load modules if required (e.g., for anvil, uncomment below)
# module load anaconda
# module load cuda/11.7

# Activate environment
source ~/.bashrc
conda activate cnn-env  # Replace with your conda environment name

# Go to working directory
cd $SLURM_SUBMIT_DIR

# Create logs directory if it doesn't exist
mkdir -p logs

# Run Python script
python main.py
