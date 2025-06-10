#!/bin/bash
# FILENAME:  cnn-job-submission

#SBATCH --job-name=cnn-train
#SBATCH --output=cnn_output.txt
#SBATCH --error=cnn_error.txt
#SBATCH --partition=gpu
#SBATCH --account=cis230030-gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:30:00

source ~/.bashrc
conda activate cnn-env
cd $SLURM_SUBMIT_DIR
python 2d_cnn.py