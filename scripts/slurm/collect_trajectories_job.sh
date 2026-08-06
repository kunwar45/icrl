#!/bin/bash
# ABOUTME: SLURM wrapper for scripts/collect_trajectories.py — starts vLLM sized from the
# ABOUTME: config, then collects. Run: CONFIG=configs/trajectory_collection/<run>.yaml sbatch scripts/slurm/collect_trajectories_job.sh
#
# One wrapper for every collection run; the config decides model, tasks, and
# keep rule. GPU count on the sbatch line MUST match the config's
# model.tensor_parallel:
#
#   # Expert (Qwen-72B, tensor_parallel: 4):
#   CONFIG=configs/trajectory_collection/stwebagentbench_expert.yaml \
#     sbatch --account=$ICRL_ACCOUNT --gres=gpu:h100:4 scripts/slurm/collect_trajectories_job.sh
#
#   # Unsafe (Qwen-7B, tensor_parallel: 1):
#   CONFIG=configs/trajectory_collection/stwebagentbench_unsafe.yaml \
#     sbatch --account=$ICRL_ACCOUNT --gres=gpu:h100:1 scripts/slurm/collect_trajectories_job.sh
#
#   # Config overrides pass through:
#   OVERRIDES="benchmark.task_ids=[235,236]" CONFIG=... sbatch ...
#
#   # Validate wiring without GPUs or browser:
#   DRY_RUN=1 sbatch --gres= --mem=4G --time=00:10:00 scripts/slurm/collect_trajectories_job.sh
#
# Prerequisites (login node): SuiteCRM running (start_suitecrm_apptainer.sh),
# WA_SUITECRM in .env, model prefetched (prefetch_models.py) — compute nodes
# are offline.

#SBATCH --job-name=icrl-collect-trajectories
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/slurm/%x_%j.out
#SBATCH --error=logs/slurm/%x_%j.err

set -euo pipefail

# Slurm spools the batch script elsewhere; SLURM_SUBMIT_DIR is the repo root.
ICRL_REPO="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "${ICRL_REPO}"
# shellcheck disable=SC1091
source "${ICRL_REPO}/scripts/slurm/job_environment.sh"

CONFIG="${CONFIG:?set CONFIG=configs/trajectory_collection/<run>.yaml on the sbatch line}"
OVERRIDES="${OVERRIDES:-}"
mkdir -p logs/slurm

# Compute nodes have no internet; every model must already be in $HF_HOME.
export HF_HUB_OFFLINE=1

# ── Read model settings out of the config (single source of truth) ───────────
read -r BACKEND MODEL TP < <(python - "${CONFIG}" <<'PY'
import sys
from omegaconf import OmegaConf
m = OmegaConf.load(sys.argv[1]).model
print(m.backend, m.name, m.get("tensor_parallel", 1))
PY
)
echo "[$(date)] config=${CONFIG} backend=${BACKEND} model=${MODEL} tp=${TP}"

OVERRIDE_ARGS=()
if [ -n "${OVERRIDES}" ]; then
    for o in ${OVERRIDES}; do OVERRIDE_ARGS+=(--override "${o}"); done
fi

if [ "${DRY_RUN:-0}" = "1" ]; then
    python scripts/collect_trajectories.py --config "${CONFIG}" "${OVERRIDE_ARGS[@]}" --dry-run
    exit 0
fi

# ── vLLM server (only when the config asks for it) ───────────────────────────
VLLM_PID=""
if [ "${BACKEND}" = "vllm" ]; then
    echo "[$(date)] starting vLLM: ${MODEL} (tensor-parallel=${TP})"
    VLLM_LOG="logs/slurm/vllm_${SLURM_JOB_ID:-local}.log"
    VLLM_USE_FLASHINFER_SAMPLER=0 python -m vllm.entrypoints.openai.api_server \
        --model "${MODEL}" \
        --tensor-parallel-size "${TP}" \
        --port 8000 \
        --max-model-len 16384 \
        > "${VLLM_LOG}" 2>&1 &
    VLLM_PID=$!

    WAITED=0
    until curl -sf http://localhost:8000/health > /dev/null 2>&1; do
        if ! kill -0 "${VLLM_PID}" 2>/dev/null; then
            echo "ERROR: vLLM died during startup" >&2
            tail -30 "${VLLM_LOG}" >&2
            exit 1
        fi
        sleep 10; WAITED=$((WAITED + 10))
        if [ "${WAITED}" -ge 900 ]; then
            echo "ERROR: vLLM not up after ${WAITED}s" >&2
            tail -30 "${VLLM_LOG}" >&2
            exit 1
        fi
    done
    echo "[$(date)] vLLM ready after ${WAITED}s"
    trap '[ -n "${VLLM_PID}" ] && kill "${VLLM_PID}" 2>/dev/null || true' EXIT
fi

# ── Collect ──────────────────────────────────────────────────────────────────
python scripts/collect_trajectories.py --config "${CONFIG}" "${OVERRIDE_ARGS[@]}"
echo "[$(date)] done."
