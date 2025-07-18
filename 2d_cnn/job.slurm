#!/bin/bash
# FILENAME:  dense-net

#SBATCH --job-name=dense-net
#SBATCH --output=logs/cnn_%j.out
#SBATCH --error=logs/cnn_%j.err
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
python dense-net.py