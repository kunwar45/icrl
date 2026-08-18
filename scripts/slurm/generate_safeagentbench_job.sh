#!/bin/bash
# ABOUTME: Cluster job for the SafeAgentBench contrast dataset — one vLLM server, expert then unsafe
# ABOUTME: Run: sbatch --account=$ICRL_ACCOUNT --gres=gpu:l40s:4 scripts/slurm/generate_safeagentbench_job.sh
#
# Differs from the ST-WebAgentBench wrapper in three ways, all because the
# environment is a simulator rather than a shared web app:
#
#   * no application lock — every episode launches its own AI2-THOR process and
#     resets its own scene, so two tasks cannot see each other's state;
#   * no reseeding — the scene reset IS the reseed, per episode, for free;
#   * vLLM is capped below its default memory share so Unity has somewhere to
#     render. The 72B needs ~136GB of the 192GB on 4xL40S; each concurrent
#     episode wants ~1-2GB more for a 300x300 framebuffer.
#
# Prerequisites (login node): the 797MB AI2-THOR CloudRendering build prefetched
# (compute nodes have no internet), SafeAgentBench cloned, and the model in
# $HF_HOME.

#SBATCH --job-name=icrl-safeagentbench
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=logs/slurm/%x_%j.out
#SBATCH --error=logs/slurm/%x_%j.err

set -euo pipefail

ICRL_REPO="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "${ICRL_REPO}"
# shellcheck disable=SC1091
source "${ICRL_REPO}/scripts/slurm/job_environment.sh"

CONFIG="${CONFIG:-configs/trajectory_generation/safeagentbench_contrast.yaml}"
SETS="${SETS:-expert,unsafe}"
OVERRIDES="${OVERRIDES:-}"
CYCLES="${CYCLES:-1}"
SAFEAGENTBENCH_ROOT="${SAFEAGENTBENCH_ROOT:-/project/aip-s2ganapa/kunwar/SafeAgentBench}"
# LowLevelPlanner is imported as `low_level_controller.low_level_controller`,
# which only resolves with the benchmark repo on the path.
export PYTHONPATH="${SAFEAGENTBENCH_ROOT}:${PYTHONPATH:-}"
export SAFEAGENTBENCH_ROOT
export HF_HUB_OFFLINE=1
mkdir -p logs/slurm

echo "[$(date)] node $(hostname)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader | head -4

read -r BACKEND MODEL TP URL < <(python - "${CONFIG}" <<'PY'
import sys
from omegaconf import OmegaConf
cfg = OmegaConf.load(sys.argv[1])
p, e = cfg.models.planner, cfg.models.executor
if (p.backend, p.name, p.get("vllm_url", "-")) != (e.backend, e.name, e.get("vllm_url", "-")):
    sys.exit("planner and executor must be the same model for a contrast dataset")
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
        --sets "${SETS}" ${OVERRIDE_ARGS[@]+"${OVERRIDE_ARGS[@]}"} --dry-run
    exit 0
fi

# ── vLLM ─────────────────────────────────────────────────────────────────────
if [ "${BACKEND}" = "vllm" ]; then
    PORT="$(echo "${URL}" | sed -E 's|.*:([0-9]+).*|\1|')"
    VLLM_LOG="logs/slurm/vllm_${SLURM_JOB_ID:-local}.log"
    echo "[$(date)] starting vLLM: ${MODEL} tp=${TP} port=${PORT}"
    CUDA_VISIBLE_DEVICES="$(seq -s, 0 $((TP - 1)))" VLLM_USE_FLASHINFER_SAMPLER=0 \
        python -m vllm.entrypoints.openai.api_server \
        --model "${MODEL}" --tensor-parallel-size "${TP}" \
        --port "${PORT}" --max-model-len 8192 \
        --gpu-memory-utilization "${VLLM_GPU_FRACTION:-0.80}" \
        --enable-prefix-caching \
        > "${VLLM_LOG}" 2>&1 &
    VLLM_PID=$!
    ICRL_CLEANUP_PIDS+=("${VLLM_PID}")

    WAITED=0
    until curl -sf "${URL%/v1}/health" > /dev/null 2>&1; do
        if ! kill -0 "${VLLM_PID}" 2>/dev/null; then
            echo "ERROR: vLLM died during startup" >&2; tail -30 "${VLLM_LOG}" >&2; exit 1
        fi
        sleep 10; WAITED=$((WAITED + 10))
        if [ "${WAITED}" -ge 1200 ]; then echo "ERROR: vLLM not up after ${WAITED}s" >&2; exit 1; fi
    done
    echo "[$(date)] vLLM ready after ${WAITED}s"
fi

# ── Generate ─────────────────────────────────────────────────────────────────
# The per-task target counts traces already on disk, so cycles converge on it
# and a task already at target is skipped without launching a simulator.
CYCLES_KEPT=0
for cycle in $(seq 1 "${CYCLES}"); do
    echo "[$(date)] ── cycle ${cycle}/${CYCLES} ──"
    if python scripts/generate_contrast_dataset.py --config "${CONFIG}" \
           --sets "${SETS}" ${OVERRIDE_ARGS[@]+"${OVERRIDE_ARGS[@]}"}; then
        CYCLES_KEPT=$((CYCLES_KEPT + 1))
    else
        echo "[$(date)] cycle ${cycle} left a half empty — continuing"
    fi
done

echo "[$(date)] done: ${CYCLES_KEPT}/${CYCLES} cycles kept traces"
python - "${CONFIG}" <<'PY'
import sys
from pathlib import Path
from omegaconf import OmegaConf
cfg = OmegaConf.to_container(OmegaConf.load(sys.argv[1]), resolve=True)
target = int(cfg["generation_loop"]["traces_per_task"])
for name in ("expert", "unsafe"):
    d = Path(cfg["sets"][name]["output"]["dir"])
    counts = {}
    for p in d.glob("task_*_trace_*.json"):
        t = p.stem.split("_")[1]
        counts[t] = counts.get(t, 0) + 1
    at = sum(1 for n in counts.values() if n >= target)
    print(f"{name:8s} {sum(counts.values()):4d} traces  {len(counts)} task(s)  {at} at {target}/task")
PY

if [ "${CYCLES_KEPT}" -eq 0 ]; then
    echo "ERROR: no cycle kept a single trace" >&2
    exit 1
fi
