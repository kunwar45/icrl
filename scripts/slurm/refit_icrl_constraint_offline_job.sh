#!/bin/bash
# ABOUTME: SLURM wrapper for scripts/refit_icrl_constraint_offline.py: replays a finished ICRL run's per-round policy samples through the published and calibrated PUCL rules (no policy training)
# ABOUTME: Submit from the repo root with CONFIG=<calibrated config> RUN_DIR=<finished ICRL run> exported, --gres=gpu:h100:4 and --export=ALL (SMOKE=1 for 2 rounds; QUANTILES=0.95,0.99,0.999)
#SBATCH --job-name=refit-icrl-constraint-offline
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --time=3:00:00
#SBATCH --output=logs/slurm/%x_%j.out
#SBATCH --error=logs/slurm/%x_%j.err
set -euo pipefail
ICRL_REPO="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}"
cd "${ICRL_REPO}"
source scripts/slurm/job_environment.sh
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_USE_FLASHINFER_SAMPLER=0
: "${CONFIG:?set CONFIG to the calibrated ICRL config}"
: "${RUN_DIR:?set RUN_DIR to the finished ICRL run directory}"
echo "=== refit-icrl-constraint-offline === job ${SLURM_JOB_ID:-local} node $(hostname) config ${CONFIG} run ${RUN_DIR} smoke=${SMOKE:-0}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
python scripts/refit_icrl_constraint_offline.py --config "${CONFIG}" --run-dir "${RUN_DIR}" \
    --quantiles "${QUANTILES:-0.95,0.99,0.999}" ${SMOKE:+--smoke}
