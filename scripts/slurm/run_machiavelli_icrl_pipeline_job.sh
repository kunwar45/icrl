#!/bin/bash
# ABOUTME: SLURM wrapper for scripts/run_machiavelli_icrl_pipeline.py — the whole MACHIAVELLI ICRL run (demos, constraint, DPO rounds, evaluation) in one resumable job
# ABOUTME: Submit from the repo root: sbatch --account=$ICRL_ACCOUNT --gres=gpu:l40s:4 --export=ALL scripts/slurm/run_machiavelli_icrl_pipeline_job.sh  (SMOKE=1 for the smoke run)
#SBATCH --job-name=machiavelli-icrl
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --time=10:00:00
#SBATCH --output=logs/slurm/%x_%j.out
#SBATCH --error=logs/slurm/%x_%j.err
set -euo pipefail
ICRL_REPO="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}"
cd "${ICRL_REPO}"
source scripts/slurm/job_environment.sh
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
: "${MACHIAVELLI_ROOT:?set MACHIAVELLI_ROOT to the benchmark clone with game_data/}"
CONFIG="${CONFIG:-configs/lagrangian_finetuning/machiavelli_constrained_dpo.yaml}"
echo "=== machiavelli-icrl === job ${SLURM_JOB_ID:-local} node $(hostname) config ${CONFIG} commit $(git rev-parse --short HEAD) smoke=${SMOKE:-0}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
python scripts/run_machiavelli_icrl_pipeline.py --config "${CONFIG}" ${SMOKE:+--smoke} ${EXTRA:-}
echo "=== DONE ==="
