#!/bin/bash
#SBATCH --job-name=icrl-constraint
#SBATCH --account=def-s2ganapa
#SBATCH --nodes=1
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=logs/slurm/%x_%j.out
#SBATCH --error=logs/slurm/%x_%j.err

set -euo pipefail
source "$(dirname "$0")/env.sh"
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

python scripts/train_constraint.py "$@"
