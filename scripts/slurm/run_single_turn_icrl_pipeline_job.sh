#!/bin/bash
# ABOUTME: SLURM wrapper for scripts/run_single_turn_icrl_pipeline.py: one Countdown-Code stage set (SFT priming, or one arm with its evaluation) in one job
# ABOUTME: Submit from the repo root with CONFIG=configs/lagrangian_finetuning/countdown_code_<arm>_qwen7b.yaml exported, --gres=gpu:h100:4 and --export=ALL (SMOKE=1 smoke run; STAGES=... a subset)
#SBATCH --job-name=single-turn-icrl
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --time=12:00:00
#SBATCH --output=logs/slurm/%x_%j.out
#SBATCH --error=logs/slurm/%x_%j.err
set -euo pipefail
ICRL_REPO="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}"
cd "${ICRL_REPO}"
source scripts/slurm/job_environment.sh
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_USE_FLASHINFER_SAMPLER=0
: "${CONFIG:?set CONFIG to a configs/lagrangian_finetuning/countdown_code_*.yaml file}"
echo "=== single-turn-icrl === job ${SLURM_JOB_ID:-local} node $(hostname) config ${CONFIG} commit $(git rev-parse --short HEAD 2>/dev/null || echo uncommitted-copy) smoke=${SMOKE:-0} stages=${STAGES:-default}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
python scripts/run_single_turn_icrl_pipeline.py --config "${CONFIG}" ${SMOKE:+--smoke} ${STAGES:+--stages "$STAGES"} ${EXTRA:-}
