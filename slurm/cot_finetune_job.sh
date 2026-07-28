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
# Slurm copies the batch script into a spool directory before running it, so
# "$(dirname "$0")" points at /cm/local/.../spool/job<N>/ and not at the repo.
# SLURM_SUBMIT_DIR is the directory sbatch was invoked from — the repo root.
ICRL_REPO="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "${ICRL_REPO}"
source "${ICRL_REPO}/slurm/env.sh"

python scripts/cot_finetune.py "$@"
