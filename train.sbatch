#!/bin/bash
#SBATCH --job-name=cs173-grpo
#SBATCH --account=scholar
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=cs173-grpo-%j.log
set -euo pipefail
cd "${PROJECT_DIR:-$HOME/CS173-GRPO}"
bash run_on_allocated_gpu.sh
