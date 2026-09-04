#!/bin/bash
# ABOUTME: SLURM wrapper for scripts/finetune_policy_dpo.py — the ODCV DPO baseline on a multi-GPU node
# ABOUTME: Submit from the repo root: CONFIG=<yaml> sbatch --account=$ICRL_ACCOUNT --gres=gpu:l40s:4 --export=ALL scripts/slurm/finetune_policy_dpo_job.sh
#SBATCH --job-name=icrl-dpo
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=06:00:00
#SBATCH --output=logs/slurm/%x_%j.out
#SBATCH --error=logs/slurm/%x_%j.err
set -euo pipefail
ICRL_REPO="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}"
cd "${ICRL_REPO}"
source scripts/slurm/job_environment.sh
export HF_HUB_OFFLINE=1
CONFIG="${CONFIG:-configs/lagrangian_finetuning/odcv_dpo_baseline.yaml}"
echo "=== icrl-dpo === job ${SLURM_JOB_ID:-local} node $(hostname) config ${CONFIG} commit $(git rev-parse --short HEAD)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
python scripts/finetune_policy_dpo.py --config "${CONFIG}" ${EXTRA:-}
echo "=== DONE ==="
