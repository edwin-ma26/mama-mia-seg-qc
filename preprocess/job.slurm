#!/bin/bash
#SBATCH --job-name=preprocess_mri
#SBATCH --output=logs/preprocess_%j.out
#SBATCH --error=logs/preprocess_%j.err
#SBATCH --account=cis230030            # Replace if needed
#SBATCH --cpus-per-task=8              # Number of CPU cores to use
#SBATCH --mem=64G                      # Adjust memory as needed
#SBATCH --time=6:00:00                # Adjust time limit
#SBATCH --nodes=1
#SBATCH --ntasks=1

# Load any necessary modules
# module load anaconda

# Activate your conda environment
source ~/.bashrc
conda activate cnn-env  # Replace with your environment name

# Go to your working directory
cd $SLURM_SUBMIT_DIR

# Create logs directory if it doesn't exist
mkdir -p logs

# Run the script
python preprocess_train_test.py
