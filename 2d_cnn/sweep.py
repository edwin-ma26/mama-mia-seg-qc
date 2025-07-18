import os
import sys
import subprocess
from multiprocessing import Pool
import itertools

# -------------------------------
# Define your reduced sweep here
# -------------------------------
PARAMETER_SWEEP = {
    'ENTROPY_REG_WEIGHT': [0.0001, 0.001, 0.005],
    'attn_dropout': [0.05, 0.1, 0.2],
    'weight_decay': [1e-4, 1e-3],
    'temperature': [0.5, 1.0, 1.5]
}

param_names = list(PARAMETER_SWEEP.keys())
param_values = list(PARAMETER_SWEEP.values())
param_combinations = list(itertools.product(*param_values))

print(f"Total combinations: {len(param_combinations)}")

# -------------------------------
# Build commands for each combo
# -------------------------------
param_commands = []
for combo_idx, combo in enumerate(param_combinations):
    entropy_reg_weight, attn_dropout, weight_decay, temperature = combo
    cmd = [
        sys.executable,  # safer than 'python'
        "single_run.py",
        f"--entropy_reg_weight={entropy_reg_weight}",
        f"--attn_dropout={attn_dropout}",
        f"--weight_decay={weight_decay}",
        f"--temperature={temperature}",
        f"--combo_idx={combo_idx}"
    ]
    param_commands.append(cmd)

# -------------------------------
# Function to run one sweep task
# -------------------------------
def run_sweep(args):
    gpu_id, cmd = args
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    try:
        subprocess.run(cmd, env=env, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Run failed on GPU {gpu_id} with command: {' '.join(cmd)}")
        print(f"Error: {e}")

# -------------------------------
# Select GPUs you want to use
# -------------------------------
gpu_ids = [0, 1]  # adjust to your system

# -------------------------------
# Build task list and skip existing
# -------------------------------
tasks = []
for i, cmd in enumerate(param_commands):
    combo_dir = os.path.join("./output", f"param_combo_{i+1:03d}")
    if os.path.exists(combo_dir):
        print(f"✅ Skipping combo {i+1}: output folder exists → {combo_dir}")
        continue
    gpu_id = gpu_ids[i % len(gpu_ids)]
    tasks.append((gpu_id, cmd))

print(f"Number of new tasks to run: {len(tasks)}")

# -------------------------------
# Run tasks with up to N GPUs in parallel
# -------------------------------
if __name__ == "__main__":
    if tasks:
        with Pool(processes=len(gpu_ids)) as pool:
            pool.map(run_sweep, tasks)
    else:
        print("🎉 Nothing to run — all sweeps are done.")
