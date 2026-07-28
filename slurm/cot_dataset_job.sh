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
# Slurm copies the batch script into a spool directory before running it, so
# "$(dirname "$0")" points at /cm/local/.../spool/job<N>/ and not at the repo.
# SLURM_SUBMIT_DIR is the directory sbatch was invoked from — the repo root.
ICRL_REPO="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "${ICRL_REPO}"
source "${ICRL_REPO}/slurm/env.sh"

python scripts/cot_build_dataset.py "$@"
