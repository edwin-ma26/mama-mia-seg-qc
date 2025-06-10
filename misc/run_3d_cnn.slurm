#!/bin/bash
#SBATCH --job-name=mri-ddp-train
#SBATCH --output=logs/output_%j.txt
#SBATCH --error=logs/error_%j.txt

#SBATCH --partition=gpu
#SBATCH --account=cis230030-gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=160G
#SBATCH --time=70:00:00  # Estimated max time
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=your_email@domain.com

# Load environment
source ~/.bashrc
conda activate cnn-env

cd $SLURM_SUBMIT_DIR
mkdir -p logs

# Run using PyTorch DDP
python -m torch.distributed.run --nproc_per_node=4 your_script.py
