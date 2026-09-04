#!/bin/bash
# ABOUTME: SLURM wrapper for scripts/finetune_policy_lagrangian_rejection_sampling.py — one round per job; chain rounds with --dependency=afterok
# ABOUTME: Submit from the repo root: sbatch --account=$ICRL_ACCOUNT --gres=gpu:l40s:4 --export=ALL scripts/slurm/finetune_policy_lagrangian_rejection_sampling_job.sh
#
# Five rounds, chained:
#   prev=$(sbatch --parsable ... this_job.sh); for i in 1 2 3 4; do prev=$(sbatch --parsable --dependency=afterok:$prev ... this_job.sh); done
# The script reads the next round from <output.dir>/state.json, so every job is the same command.
#SBATCH --job-name=odcv-lagrangian-rs
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --time=04:00:00
#SBATCH --output=logs/slurm/%x_%j.out
#SBATCH --error=logs/slurm/%x_%j.err
set -euo pipefail
ICRL_REPO="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}"
cd "${ICRL_REPO}"
source scripts/slurm/job_environment.sh
module load apptainer/1.4.5 2>/dev/null || module load apptainer 2>/dev/null || true
export HF_HUB_OFFLINE=1
CONFIG="${CONFIG:-configs/lagrangian_finetuning/odcv_lagrangian_rejection_sampling.yaml}"
echo "=== odcv-lagrangian-rs === job ${SLURM_JOB_ID:-local} node $(hostname) config ${CONFIG} commit $(git rev-parse --short HEAD)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
apptainer --version
python scripts/finetune_policy_lagrangian_rejection_sampling.py --config "${CONFIG}" ${EXTRA:-}
echo "=== DONE ==="
