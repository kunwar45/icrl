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
#     sbatch --account=$ICRL_ACCOUNT --gres=gpu:h100:4 scripts/slurm/generate_trajectories_job.sh
#
#   # Separate executor model instead (e.g. in-distribution 7B → 5 GPUs total):
#   OVERRIDES="models.executor.name=Qwen/Qwen2.5-7B-Instruct models.executor.vllm_url=http://localhost:8001/v1 models.executor.tensor_parallel=1" \
#     CONFIG=... sbatch --account=$ICRL_ACCOUNT --gres=gpu:h100:5 scripts/slurm/generate_trajectories_job.sh
#
#   # Wiring check, no GPUs or browser:
#   DRY_RUN=1 sbatch --gres= --mem=4G --time=00:10:00 scripts/slurm/generate_trajectories_job.sh
#
# Prerequisites (login node): SuiteCRM running (start_suitecrm_apptainer.sh),
# WA_SUITECRM in .env, BOTH models prefetched (prefetch_models.py) — compute
# nodes are offline.

#SBATCH --job-name=icrl-generate-trajectories
#SBATCH --nodes=1
# Episodes run concurrently (generation_loop.concurrency), and each concurrent
# episode is a headless Chromium plus Python-side axtree flattening — CPU, not
# GPU, is what limits how many can be in flight. Keep this at roughly twice the
# config's concurrency.
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=logs/slurm/%x_%j.out
#SBATCH --error=logs/slurm/%x_%j.err

set -euo pipefail

ICRL_REPO="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "${ICRL_REPO}"
# shellcheck disable=SC1091
source "${ICRL_REPO}/scripts/slurm/job_environment.sh"

CONFIG="${CONFIG:?set CONFIG=configs/trajectory_generation/<run>.yaml on the sbatch line}"
OVERRIDES="${OVERRIDES:-}"
mkdir -p logs/slurm

export HF_HUB_OFFLINE=1

# ── Mutual exclusion against the shared web app ──────────────────────────────
# The lock, the shard resolution and the single EXIT handler all live in
# job_environment.sh so the collection wrapper enforces exactly the same rule —
# an unsafe collection pass mutating SuiteCRM underneath a generation pass
# poisons its traces just as surely as a second generation pass would.
if ! icrl_take_suitecrm_lock; then
    echo "[$(date)] another pass holds ${ICRL_SUITECRM_LOCK} — exiting so it can finish"
    exit 0
fi
echo "[$(date)] holding SuiteCRM lock ${ICRL_SUITECRM_LOCK}"

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
        --enable-prefix-caching \
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

if [ "${P_BACKEND}" = "vllm" ]; then
    if [ "${E_BACKEND}" = "vllm" ] && { [ "${E_MODEL}" != "${P_MODEL}" ] || [ "${E_URL}" != "${P_URL}" ]; }; then
        # Two servers on disjoint GPU ranges: planner gets 0..P_TP-1, executor the rest.
        P_GPUS=$(seq -s, 0 $((P_TP - 1)))
        E_GPUS=$(seq -s, "${P_TP}" $((P_TP + E_TP - 1)))
        P_PID=$(start_vllm "${P_MODEL}" "${P_TP}" "${P_URL}" "${P_GPUS}" planner);  ICRL_CLEANUP_PIDS+=("${P_PID}")
        E_PID=$(start_vllm "${E_MODEL}" "${E_TP}" "${E_URL}" "${E_GPUS}" executor); ICRL_CLEANUP_PIDS+=("${E_PID}")
        wait_vllm "${P_URL}" "${P_PID}" planner
        wait_vllm "${E_URL}" "${E_PID}" executor
    else
        # One shared server (same model+url, or executor not on vLLM).
        P_GPUS=$(seq -s, 0 $((P_TP - 1)))
        P_PID=$(start_vllm "${P_MODEL}" "${P_TP}" "${P_URL}" "${P_GPUS}" planner); ICRL_CLEANUP_PIDS+=("${P_PID}")
        wait_vllm "${P_URL}" "${P_PID}" planner
    fi
fi

# ── Generate ─────────────────────────────────────────────────────────────────
# RESEED_BEFORE_RUN=1 resets the demo data before each cycle: a trace is kept
# only if the database shows the change AND did not already show it beforehand,
# so a cycle over state left by the previous one can neither succeed nor fail
# honestly (a delete has no target left; created records already exist). Safe
# only because the lock above serialises passes — this wipes the demo records
# any concurrent job would be reading.
#
# CYCLES reseed-and-generate rounds inside ONE job. Loading the 72B costs ~130s
# and re-queueing costs minutes-to-hours, so a job that only does one pass spends
# most of its life on startup. Traces accumulate as trace_0, trace_1, ... across
# cycles, exactly as they would across separate jobs.
#
# A BARREN CYCLE MUST NOT KILL THE CHAIN. generate_trajectories.py exits
# non-zero when a pass keeps nothing — deliberately, so a one-shot job is marked
# failed rather than leaving an empty directory that looks finished. Under
# `set -e` that turned the first barren cycle into the end of the run: job
# 4806706 asked for 70 cycles, kept nothing in cycle 1, and died 35 minutes in
# having produced no traces. Cycles are independent attempts over a freshly
# reseeded database, and an empty one is an ordinary outcome, not an error.
# The job still fails if EVERY cycle came up empty, which is the condition the
# non-zero exit was actually there to signal.
CYCLES_KEPT=0
CYCLES_EMPTY=0
for cycle in $(seq 1 "${CYCLES:-1}"); do
    echo "[$(date)] ── cycle ${cycle}/${CYCLES:-1} ──"
    if [ "${RESEED_BEFORE_RUN:-0}" = "1" ]; then
        echo "[$(date)] reseeding SuiteCRM demo data"
        bash scripts/reseed_suitecrm_demo_data.sh > /dev/null
    fi
    if python scripts/generate_trajectories.py --config "${CONFIG}" "${OVERRIDE_ARGS[@]}"; then
        CYCLES_KEPT=$((CYCLES_KEPT + 1))
    else
        CYCLES_EMPTY=$((CYCLES_EMPTY + 1))
        echo "[$(date)] cycle ${cycle} kept nothing — continuing to the next cycle"
    fi
done
echo "[$(date)] done: ${CYCLES_KEPT} cycles kept traces, ${CYCLES_EMPTY} came up empty."
if [ "${CYCLES_KEPT}" -eq 0 ]; then
    echo "ERROR: no cycle kept a single trace" >&2
    exit 1
fi
