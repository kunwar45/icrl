#!/bin/bash
#SBATCH --job-name=icrl-cot-dataset
#SBATCH --account=def-s2ganapa
#SBATCH --nodes=1
#SBATCH --gres=gpu:h100:1          # CausalScorer requires a GPU for forward-pass hooks
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=logs/slurm/%x_%j.out
#SBATCH --error=logs/slurm/%x_%j.err

set -euo pipefail
source "$(dirname "$0")/env.sh"
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

python scripts/cot_build_dataset.py "$@"
