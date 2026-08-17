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
#   # Several reseed-and-collect cycles in one allocation (150 unsafe demos):
#   CYCLES=4 RESEED_BEFORE_RUN=1 \
#     CONFIG=configs/trajectory_collection/stwebagentbench_unsafe.yaml \
#     sbatch --account=$ICRL_ACCOUNT --gres=gpu:h100:1 scripts/slurm/collect_trajectories_job.sh
#
#   # Against SuiteCRM shard 2 instead of the single shared instance:
#   ICRL_SUITECRM_SHARD=2 CONFIG=... sbatch ...
#
#   # Validate wiring without GPUs or browser:
#   DRY_RUN=1 sbatch --gres= --mem=4G --time=00:10:00 scripts/slurm/collect_trajectories_job.sh
#
# Prerequisites (login node): SuiteCRM running (start_suitecrm_apptainer.sh, or
# start_suitecrm_shards.sh when sharding), WA_SUITECRM in .env, model prefetched
# (prefetch_models.py) — compute nodes are offline.
#
# This wrapper takes the same SuiteCRM lock as the generation wrapper: a
# collection pass mutates the CRM, so overlapping one with a generation pass
# corrupts that pass's database verdicts.

#SBATCH --job-name=icrl-collect-trajectories
#SBATCH --nodes=1
# Episodes run concurrently (episode.concurrency), and each concurrent episode
# is a headless Chromium plus Python-side axtree flattening — CPU, not GPU, is
# what limits how many can be in flight. Keep this at roughly twice the
# config's concurrency.
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
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

# ── Mutual exclusion against the shared web app ──────────────────────────────
# Collection mutates SuiteCRM exactly as much as generation does — an unsafe
# episode really deletes the lead. Running one alongside a generation pass
# changes the database inside that pass's before/after comparison and silently
# poisons its expert traces (they look verified and prove nothing). This
# wrapper used to take no lock at all, so the two could and did overlap.
# Same helper, same lock, so the two kinds of pass exclude each other.
if ! icrl_take_suitecrm_lock; then
    echo "[$(date)] another pass holds ${ICRL_SUITECRM_LOCK} — exiting so it can finish"
    exit 0
fi
echo "[$(date)] holding SuiteCRM lock ${ICRL_SUITECRM_LOCK}"

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
    # Prefix caching matters here as much as in generation: the configs put the
    # static action space in the system message and the volatile axtree last, so
    # consecutive steps of an episode share a long prefix.
    VLLM_USE_FLASHINFER_SAMPLER=0 python -m vllm.entrypoints.openai.api_server \
        --model "${MODEL}" \
        --tensor-parallel-size "${TP}" \
        --port 8000 \
        --max-model-len 16384 \
        --enable-prefix-caching \
        > "${VLLM_LOG}" 2>&1 &
    VLLM_PID=$!
    # Registered on the SHARED cleanup list, not a fresh `trap ... EXIT`: bash
    # keeps one EXIT trap, and a second one here would silently replace the
    # lock's handler and leave the lock directory behind for every later pass.
    ICRL_CLEANUP_PIDS+=("${VLLM_PID}")

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
fi

# ── Collect ──────────────────────────────────────────────────────────────────
# CYCLES reseed-and-collect rounds inside ONE job, as in the generation wrapper:
# model load plus queue time dwarfs a single pass, and traces accumulate as
# trace_0, trace_1, ... across cycles exactly as they would across jobs.
#
# RESEED_BEFORE_RUN=1 matters for a different reason here than in generation.
# Nothing in the unsafe keep rule reads the database, so a stale database does
# not make a trace WRONG — it makes it uninteresting. After the first pass has
# deleted the lead and closed the case, later rollouts wander among records that
# no longer exist, and "agent confused by missing data" is not the unsafe
# behaviour this set is supposed to demonstrate. Reseeding between cycles keeps
# every rollout starting from the same reachable world.
# A barren cycle must not kill the chain — see the same guard in
# generate_trajectories_job.sh. The collector exits non-zero when a pass keeps
# nothing, and under `set -e` that ends the whole run on the first empty cycle
# instead of trying again over a freshly reseeded database.
CYCLES_KEPT=0
CYCLES_EMPTY=0
for cycle in $(seq 1 "${CYCLES:-1}"); do
    echo "[$(date)] ── cycle ${cycle}/${CYCLES:-1} ──"
    if [ "${RESEED_BEFORE_RUN:-0}" = "1" ]; then
        echo "[$(date)] reseeding SuiteCRM demo data"
        bash scripts/reseed_suitecrm_demo_data.sh > /dev/null
    fi
    if python scripts/collect_trajectories.py --config "${CONFIG}" "${OVERRIDE_ARGS[@]}"; then
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
