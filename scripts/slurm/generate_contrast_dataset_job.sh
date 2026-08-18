#!/bin/bash
# ABOUTME: THE cluster job for the ICRL contrast dataset — one vLLM server, expert then unsafe,
# ABOUTME: reseeded cycles until every task hits its target. Run: sbatch scripts/slurm/generate_contrast_dataset_job.sh
#
# This is the whole data stage in one submission. It starts the 72B, generates
# the expert set, generates the unsafe set against the same tasks with the same
# model, reseeds, and repeats until every task has its traces_per_task or the
# wall clock runs out.
#
#   sbatch --account=$ICRL_ACCOUNT --gres=gpu:h100:4 \
#       scripts/slurm/generate_contrast_dataset_job.sh
#
#   # Wiring check, no GPUs or browser:
#   DRY_RUN=1 sbatch --gres= --mem=4G --time=00:10:00 \
#       scripts/slurm/generate_contrast_dataset_job.sh
#
# Prerequisites (login node): SuiteCRM running (start_suitecrm_apptainer.sh),
# WA_SUITECRM and ICRL_SUITECRM_DB_PASSWORD in .env, the 72B prefetched
# (prefetch_models.py) — compute nodes are offline.
#
# WHY ONE JOB AND NOT TWO. Both sets run the same Qwen2.5-72B, so the server is
# loaded once (~130s) and serves both halves instead of two jobs each paying
# load plus queue time. It is also the only way to guarantee the two halves saw
# the same environment: an unsafe pass really does delete the record an expert
# pass is verified against, so they are serialised here rather than racing as
# separate submissions.

#SBATCH --job-name=icrl-generate-contrast
#SBATCH --nodes=1
# Episodes run concurrently (generation_loop.concurrency), and each concurrent
# episode is a headless Chromium plus Python-side axtree flattening — CPU, not
# GPU, is what limits how many can be in flight. Roughly twice the config's
# concurrency.
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
# Sized to the target: this is meant to finish the data stage in one sitting.
# Cycles stop early when the remaining time cannot fit another one, so the job
# always exits cleanly with its traces rather than being killed mid-episode.
#SBATCH --time=04:00:00
#SBATCH --output=logs/slurm/%x_%j.out
#SBATCH --error=logs/slurm/%x_%j.err

set -euo pipefail

ICRL_REPO="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "${ICRL_REPO}"
# shellcheck disable=SC1091
source "${ICRL_REPO}/scripts/slurm/job_environment.sh"

CONFIG="${CONFIG:-configs/trajectory_generation/stwebagentbench_contrast.yaml}"
OVERRIDES="${OVERRIDES:-}"
# Both halves by default. Narrow it only to resume a run whose other half is
# already at target — a dataset with one half is not a dataset.
SETS="${SETS:-expert,unsafe}"
# Reseeding is ON by default here, unlike the single-set wrapper. A trace is kept
# only if the database shows the change AND did not already show it beforehand,
# so a cycle run over the state the previous cycle left can neither succeed nor
# fail honestly — and the unsafe half of each cycle deliberately breaks things.
RESEED_BEFORE_RUN="${RESEED_BEFORE_RUN:-1}"
# Upper bound on cycles; the wall-clock guard below is what usually stops it.
CYCLES="${CYCLES:-12}"
# Stop starting new cycles when less than this many seconds remain, so the last
# cycle finishes and writes its summary instead of being killed mid-episode.
CYCLE_RESERVE_SECONDS="${CYCLE_RESERVE_SECONDS:-1500}"
mkdir -p logs/slurm

export HF_HUB_OFFLINE=1

# ── Mutual exclusion against the shared web app ──────────────────────────────
if ! icrl_take_suitecrm_lock; then
    echo "[$(date)] another pass holds ${ICRL_SUITECRM_LOCK} — exiting so it can finish"
    exit 0
fi
echo "[$(date)] holding SuiteCRM lock ${ICRL_SUITECRM_LOCK}"

# ── Read the model block (both sets share it by construction) ────────────────
read -r BACKEND MODEL TP URL < <(python - "${CONFIG}" <<'PY'
import sys
from omegaconf import OmegaConf
cfg = OmegaConf.load(sys.argv[1])
p, e = cfg.models.planner, cfg.models.executor
if (p.backend, p.name, p.get("vllm_url", "-")) != (e.backend, e.name, e.get("vllm_url", "-")):
    # Not a warning: the contrast config exists to keep the two sets identical
    # outside the experimental condition, and model identity is the confound it
    # was written to remove.
    sys.exit("planner and executor must be the same model for the contrast dataset")
print(p.backend, p.name, p.get("tensor_parallel", 1), p.get("vllm_url", "-"))
PY
)
echo "[$(date)] model=${MODEL} (tp=${TP}) serving planner and executor"

OVERRIDE_ARGS=()
if [ -n "${OVERRIDES}" ]; then
    for o in ${OVERRIDES}; do OVERRIDE_ARGS+=(--override "${o}"); done
fi

if [ "${DRY_RUN:-0}" = "1" ]; then
    python scripts/generate_contrast_dataset.py --config "${CONFIG}" \
        --sets "${SETS}" "${OVERRIDE_ARGS[@]}" --dry-run
    exit 0
fi

# ── vLLM server (one, shared by both roles and both sets) ────────────────────
if [ "${BACKEND}" = "vllm" ]; then
    PORT="$(echo "${URL}" | sed -E 's|.*:([0-9]+).*|\1|')"
    VLLM_LOG="logs/slurm/vllm_${SLURM_JOB_ID:-local}.log"
    echo "[$(date)] starting vLLM: ${MODEL} tp=${TP} port=${PORT}"
    CUDA_VISIBLE_DEVICES="$(seq -s, 0 $((TP - 1)))" VLLM_USE_FLASHINFER_SAMPLER=0 \
        python -m vllm.entrypoints.openai.api_server \
        --model "${MODEL}" --tensor-parallel-size "${TP}" \
        --port "${PORT}" --max-model-len 16384 \
        --enable-prefix-caching \
        > "${VLLM_LOG}" 2>&1 &
    VLLM_PID=$!
    ICRL_CLEANUP_PIDS+=("${VLLM_PID}")

    WAITED=0
    until curl -sf "${URL%/v1}/health" > /dev/null 2>&1; do
        if ! kill -0 "${VLLM_PID}" 2>/dev/null; then
            echo "ERROR: vLLM died during startup" >&2
            tail -30 "${VLLM_LOG}" >&2
            exit 1
        fi
        sleep 10; WAITED=$((WAITED + 10))
        if [ "${WAITED}" -ge 1200 ]; then
            echo "ERROR: vLLM not up after ${WAITED}s" >&2
            exit 1
        fi
    done
    echo "[$(date)] vLLM ready after ${WAITED}s"
fi

# ── Cycles ───────────────────────────────────────────────────────────────────
# Each cycle reseeds, then generates BOTH sets to their per-task target. The
# target counts traces already on disk, so cycles converge on a balanced set
# instead of piling more traces onto whichever task keeps most easily, and a
# task that is already done is skipped without booting a browser.
#
# A BARREN CYCLE MUST NOT KILL THE CHAIN. The script exits non-zero when a half
# keeps nothing, deliberately, so a one-shot run is marked failed rather than
# leaving a directory that looks finished. Under `set -e` that would end the run
# at the first barren cycle (it did once: 70 cycles requested, dead after one).
# An empty cycle is an ordinary outcome; the job fails only if EVERY cycle was.
CYCLES_KEPT=0
CYCLES_EMPTY=0
STARTED_AT=$(date +%s)

_seconds_left() {
    # Remaining wall clock from SLURM's own accounting, so the reserve is real
    # rather than a guess about how long the job was queued.
    local left
    left="$(squeue -h -j "${SLURM_JOB_ID:-0}" -o %L 2>/dev/null || true)"
    if [ -z "${left}" ]; then echo 999999; return; fi
    python - "${left}" <<'PY'
import sys
# SLURM prints D-HH:MM:SS, HH:MM:SS, MM:SS or SS
s = sys.argv[1].strip()
days, _, rest = s.partition("-")
if not rest:
    days, rest = "0", days
parts = [int(p) for p in rest.split(":")]
while len(parts) < 3:
    parts.insert(0, 0)
h, m, sec = parts
print(int(days) * 86400 + h * 3600 + m * 60 + sec)
PY
}

for cycle in $(seq 1 "${CYCLES}"); do
    LEFT="$(_seconds_left)"
    if [ "${LEFT}" -lt "${CYCLE_RESERVE_SECONDS}" ]; then
        echo "[$(date)] ${LEFT}s of wall clock left (< ${CYCLE_RESERVE_SECONDS}s reserve) — stopping cleanly"
        break
    fi
    echo "[$(date)] ── cycle ${cycle}/${CYCLES} (${LEFT}s left) ──"

    if [ "${RESEED_BEFORE_RUN}" = "1" ]; then
        echo "[$(date)] reseeding SuiteCRM demo data"
        bash scripts/reseed_suitecrm_demo_data.sh > /dev/null
    fi

    if python scripts/generate_contrast_dataset.py --config "${CONFIG}" \
           --sets "${SETS}" "${OVERRIDE_ARGS[@]}"; then
        CYCLES_KEPT=$((CYCLES_KEPT + 1))
    else
        CYCLES_EMPTY=$((CYCLES_EMPTY + 1))
        echo "[$(date)] cycle ${cycle} left a half empty — continuing to the next cycle"
    fi
done

ELAPSED=$(( $(date +%s) - STARTED_AT ))
echo "[$(date)] done after ${ELAPSED}s: ${CYCLES_KEPT} cycles kept traces, ${CYCLES_EMPTY} came up empty."

# Final shape report — what the next stage actually needs to know.
python - "${CONFIG}" <<'PY'
import sys
from pathlib import Path
from omegaconf import OmegaConf

cfg = OmegaConf.load(sys.argv[1])
target = int(cfg.generation_loop.traces_per_task)
for name in ("expert", "unsafe"):
    d = Path(OmegaConf.to_container(cfg, resolve=True)["sets"][name]["output"]["dir"])
    counts = {}
    for p in d.glob("task_*_trace_*.json"):
        counts.setdefault(p.stem.split("_")[1], 0)
        counts[p.stem.split("_")[1]] += 1
    total = sum(counts.values())
    at = sum(1 for n in counts.values() if n >= target)
    print(f"{name:8s} {total:4d} traces  {len(counts)} task(s)  "
          f"{at} at {target}/task  {dict(sorted(counts.items()))}")
PY

if [ "${CYCLES_KEPT}" -eq 0 ]; then
    echo "ERROR: no cycle kept a single trace" >&2
    exit 1
fi
