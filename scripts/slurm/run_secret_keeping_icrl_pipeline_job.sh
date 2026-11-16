#!/bin/bash
# ABOUTME: SLURM wrapper for scripts/run_secret_keeping_icrl_pipeline.py — the whole secret-keeping ICRL run (demos, base eval, constraint, DPO rounds, evaluation) in one resumable job
# ABOUTME: Submit from the repo root: sbatch --account=$ICRL_ACCOUNT --gres=gpu:h100:4 --export=ALL scripts/slurm/run_secret_keeping_icrl_pipeline_job.sh  (SMOKE=1 smoke run; STAGES=demos,... runs a subset)
#SBATCH --job-name=secret-keeping-icrl
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --time=06:00:00
#SBATCH --output=logs/slurm/%x_%j.out
#SBATCH --error=logs/slurm/%x_%j.err
set -euo pipefail
ICRL_REPO="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}"
cd "${ICRL_REPO}"
source scripts/slurm/job_environment.sh
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_USE_FLASHINFER_SAMPLER=0
CONFIG="${CONFIG:-configs/lagrangian_finetuning/secret_keeping_constrained_dpo_qwen7b.yaml}"
echo "=== secret-keeping-icrl === job ${SLURM_JOB_ID:-local} node $(hostname) config ${CONFIG} commit $(git rev-parse --short HEAD) smoke=${SMOKE:-0} stages=${STAGES:-all}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
python scripts/run_secret_keeping_icrl_pipeline.py --config "${CONFIG}" ${SMOKE:+--smoke} ${STAGES:+--stages "$STAGES"} ${EXTRA:-}
