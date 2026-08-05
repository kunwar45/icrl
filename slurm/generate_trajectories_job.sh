#!/bin/bash
# ABOUTME: SLURM wrapper for scripts/generate_trajectories.py — starts one or two vLLM
# ABOUTME: servers (planner + executor) sized from the config, then generates.
#
# GPU math: when planner and executor are DIFFERENT models, the job needs
# planner.tensor_parallel + executor.tensor_parallel GPUs and the wrapper starts
# two servers on disjoint CUDA_VISIBLE_DEVICES. When they are the SAME model and
# URL, one server is started and both roles share it.
#
#   # Default config (72B for planner AND executor, one shared server → 4 GPUs):
#   CONFIG=configs/trajectory_generation/stwebagentbench_expert.yaml \
#     sbatch --account=$ICRL_ACCOUNT --gres=gpu:h100:4 slurm/generate_trajectories_job.sh
#
#   # Separate executor model instead (e.g. in-distribution 7B → 5 GPUs total):
#   OVERRIDES="models.executor.name=Qwen/Qwen2.5-7B-Instruct models.executor.vllm_url=http://localhost:8001/v1 models.executor.tensor_parallel=1" \
#     CONFIG=... sbatch --account=$ICRL_ACCOUNT --gres=gpu:h100:5 slurm/generate_trajectories_job.sh
#
#   # Wiring check, no GPUs or browser:
#   DRY_RUN=1 sbatch --gres= --mem=4G --time=00:10:00 slurm/generate_trajectories_job.sh
#
# Prerequisites (login node): SuiteCRM running (start_suitecrm_apptainer.sh),
# WA_SUITECRM in .env, BOTH models prefetched (prefetch_models.py) — compute
# nodes are offline.

#SBATCH --job-name=icrl-generate-trajectories
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=logs/slurm/%x_%j.out
#SBATCH --error=logs/slurm/%x_%j.err

set -euo pipefail

ICRL_REPO="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "${ICRL_REPO}"
# shellcheck disable=SC1091
source "${ICRL_REPO}/slurm/env.sh"

CONFIG="${CONFIG:?set CONFIG=configs/trajectory_generation/<run>.yaml on the sbatch line}"
OVERRIDES="${OVERRIDES:-}"
mkdir -p logs/slurm

export HF_HUB_OFFLINE=1

# ── Read both model blocks out of the config ─────────────────────────────────
read -r P_BACKEND P_MODEL P_TP P_URL E_BACKEND E_MODEL E_TP E_URL < <(python - "${CONFIG}" <<'PY'
import sys
from omegaconf import OmegaConf
m = OmegaConf.load(sys.argv[1]).models
print(m.planner.backend, m.planner.name, m.planner.get("tensor_parallel", 1), m.planner.get("vllm_url", "-"),
      m.executor.backend, m.executor.name, m.executor.get("tensor_parallel", 1), m.executor.get("vllm_url", "-"))
PY
)
echo "[$(date)] planner=${P_MODEL} (tp=${P_TP})  executor=${E_MODEL} (tp=${E_TP})"

OVERRIDE_ARGS=()
if [ -n "${OVERRIDES}" ]; then
    for o in ${OVERRIDES}; do OVERRIDE_ARGS+=(--override "${o}"); done
fi

if [ "${DRY_RUN:-0}" = "1" ]; then
    python scripts/generate_trajectories.py --config "${CONFIG}" "${OVERRIDE_ARGS[@]}" --dry-run
    exit 0
fi

# ── vLLM servers ─────────────────────────────────────────────────────────────
_port_of() { echo "$1" | sed -E 's|.*:([0-9]+).*|\1|'; }

start_vllm() {  # start_vllm <model> <tp> <url> <gpu_list> <tag>
    local model="$1" tp="$2" url="$3" gpus="$4" tag="$5"
    local port; port="$(_port_of "${url}")"
    local log="logs/slurm/vllm_${tag}_${SLURM_JOB_ID:-local}.log"
    # stdout is captured by the caller ($(start_vllm ...) returns the PID) —
    # status lines must go to stderr.
    echo "[$(date)] starting vLLM(${tag}): ${model} tp=${tp} port=${port} gpus=${gpus}" >&2
    CUDA_VISIBLE_DEVICES="${gpus}" VLLM_USE_FLASHINFER_SAMPLER=0 \
        python -m vllm.entrypoints.openai.api_server \
        --model "${model}" --tensor-parallel-size "${tp}" \
        --port "${port}" --max-model-len 16384 \
        > "${log}" 2>&1 &
    echo $!
}

wait_vllm() {  # wait_vllm <url> <pid> <tag>
    local url="$1" pid="$2" tag="$3" waited=0
    local health="${url%/v1}/health"
    until curl -sf "${health}" > /dev/null 2>&1; do
        if ! kill -0 "${pid}" 2>/dev/null; then
            echo "ERROR: vLLM(${tag}) died during startup" >&2
            tail -30 "logs/slurm/vllm_${tag}_${SLURM_JOB_ID:-local}.log" >&2
            exit 1
        fi
        sleep 10; waited=$((waited + 10))
        if [ "${waited}" -ge 1200 ]; then
            echo "ERROR: vLLM(${tag}) not up after ${waited}s" >&2
            exit 1
        fi
    done
    echo "[$(date)] vLLM(${tag}) ready after ${waited}s"
}

PIDS=()
if [ "${P_BACKEND}" = "vllm" ]; then
    if [ "${E_BACKEND}" = "vllm" ] && { [ "${E_MODEL}" != "${P_MODEL}" ] || [ "${E_URL}" != "${P_URL}" ]; }; then
        # Two servers on disjoint GPU ranges: planner gets 0..P_TP-1, executor the rest.
        P_GPUS=$(seq -s, 0 $((P_TP - 1)))
        E_GPUS=$(seq -s, "${P_TP}" $((P_TP + E_TP - 1)))
        P_PID=$(start_vllm "${P_MODEL}" "${P_TP}" "${P_URL}" "${P_GPUS}" planner);  PIDS+=("${P_PID}")
        E_PID=$(start_vllm "${E_MODEL}" "${E_TP}" "${E_URL}" "${E_GPUS}" executor); PIDS+=("${E_PID}")
        wait_vllm "${P_URL}" "${P_PID}" planner
        wait_vllm "${E_URL}" "${E_PID}" executor
    else
        # One shared server (same model+url, or executor not on vLLM).
        P_GPUS=$(seq -s, 0 $((P_TP - 1)))
        P_PID=$(start_vllm "${P_MODEL}" "${P_TP}" "${P_URL}" "${P_GPUS}" planner); PIDS+=("${P_PID}")
        wait_vllm "${P_URL}" "${P_PID}" planner
    fi
    trap 'for p in "${PIDS[@]}"; do kill "$p" 2>/dev/null || true; done' EXIT
fi

# ── Generate ─────────────────────────────────────────────────────────────────
python scripts/generate_trajectories.py --config "${CONFIG}" "${OVERRIDE_ARGS[@]}"
echo "[$(date)] done."
