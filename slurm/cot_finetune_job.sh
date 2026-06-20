#!/bin/bash
#SBATCH --job-name=icrl-cot-finetune
#SBATCH --account=def-s2ganapa
#SBATCH --nodes=1
#SBATCH --gres=gpu:h100:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=logs/slurm/%x_%j.out
#SBATCH --error=logs/slurm/%x_%j.err

set -euo pipefail
source "$(dirname "$0")/env.sh"
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

python scripts/cot_finetune.py "$@"
