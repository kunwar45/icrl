#!/bin/bash
# ============================================================================
# gen_safe_demos.sh — Collect CuP=1 safe trajectories via vLLM + Qwen-72B
# ============================================================================
#
# Starts a vLLM server on the allocated GPUs, then runs the collection loop
# for ST-WebAgentBench Easy-tier SuiteCRM tasks (IDs 235-254 by default).
#
# ── Quick start ──────────────────────────────────────────────────────────────
#
#   # All 20 easy tasks, 5 retries each:
#   sbatch slurm/gen_safe_demos.sh
#
#   # First N tasks only (smoke test):
#   N_TASKS=3 sbatch slurm/gen_safe_demos.sh
#
#   # Explicit task IDs:
#   TASK_IDS="235 236 237 238 239" sbatch slurm/gen_safe_demos.sh
#
#   # More retries / longer episodes:
#   MAX_RETRIES=10 MAX_STEPS=40 sbatch slurm/gen_safe_demos.sh
#
#   # Dry-run (validate imports, env, vLLM health — no browser):
#   DRY_RUN=1 sbatch slurm/gen_safe_demos.sh
#
# ── Environment variables ────────────────────────────────────────────────────
#
#   N_TASKS        How many easy-tier tasks to run (default: all 20).
#                  Tasks are taken in order from EASY_TASK_IDS (235-254).
#                  Ignored if TASK_IDS is set explicitly.
#
#   TASK_IDS       Space-separated list of task IDs to run.
#                  Example: TASK_IDS="235 236 237"
#
#   MAX_RETRIES    Attempts per task before flagging as failed (default: 5).
#
#   MAX_STEPS      Max browser steps per episode (default: 30).
#
#   OUTPUT_DIR     Where to save trajectory JSON files (default: below).
#
#   MODEL          HuggingFace model ID (default: Qwen/Qwen2.5-72B-Instruct).
#
#   TP_SIZE        vLLM tensor-parallel degree, must match --gres=gpu:N
#                  (default: 4).
#
#   VLLM_PORT      Port for the vLLM OpenAI-compat server (default: 8100).
#
#   DRY_RUN        Set to 1 to run --dry-run only (no browser, no episodes).
#
#   WA_SUITECRM    SuiteCRM URL, e.g. http://localhost:8080 (required).
#                  Must be reachable from the compute node.
#
# ── Resource sizing ──────────────────────────────────────────────────────────
#
#   Qwen-72B in bfloat16 ≈ 144 GB VRAM.
#   4× A100-40GB  → set TP_SIZE=4, --gres=gpu:a100:4
#   2× A100-80GB  → set TP_SIZE=2, --gres=gpu:a100l:2   (L = 80 GB variant)
#
#SBATCH --job-name=icrl-gen
#SBATCH --account=def-srirams
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:a100:4
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=logs/slurm/%x_%j.out
#SBATCH --error=logs/slurm/%x_%j.err

set -euo pipefail

# ── Defaults (override via env vars before sbatch) ───────────────────────────

MODEL="${MODEL:-Qwen/Qwen2.5-72B-Instruct}"
TP_SIZE="${TP_SIZE:-4}"
VLLM_PORT="${VLLM_PORT:-8100}"
VLLM_URL="http://localhost:${VLLM_PORT}/v1"

MAX_RETRIES="${MAX_RETRIES:-5}"
MAX_STEPS="${MAX_STEPS:-30}"

# All 20 easy-tier task IDs in order
ALL_EASY_TASKS="235 236 237 238 239 240 241 242 243 244 245 246 247 248 249 250 251 252 253 254"

OUTPUT_DIR="${OUTPUT_DIR:-${SCRATCH:-$SLURM_SUBMIT_DIR}/trajectories/safe}"
HF_CACHE="${SCRATCH:-/tmp}/hf_cache"
LOG_DIR="logs/slurm"
DRY_RUN="${DRY_RUN:-0}"

# ── Resolve task list ─────────────────────────────────────────────────────────

if [ -n "${TASK_IDS:-}" ]; then
    # Explicit list takes priority
    RESOLVED_TASKS="${TASK_IDS}"
elif [ -n "${N_TASKS:-}" ]; then
    # Take first N from the easy-tier list
    RESOLVED_TASKS=$(echo "${ALL_EASY_TASKS}" | tr ' ' '\n' | head -n "${N_TASKS}" | tr '\n' ' ')
else
    RESOLVED_TASKS="${ALL_EASY_TASKS}"
fi

N_RESOLVED=$(echo "${RESOLVED_TASKS}" | wc -w | tr -d ' ')

# ── Setup ─────────────────────────────────────────────────────────────────────

mkdir -p "${LOG_DIR}" "${OUTPUT_DIR}" "${HF_CACHE}"

export HF_HOME="${HF_CACHE}"
export TRANSFORMERS_CACHE="${HF_CACHE}"

# Load cluster modules if available (no-op outside SLURM)
module load python/3.11 cuda/12.1 cudnn/8.9 2>/dev/null || true
source ~/envs/icrl/bin/activate 2>/dev/null || true

echo "========================================================================"
echo " icrl-gen  job ${SLURM_JOB_ID:-local}  node $(hostname)"
echo "========================================================================"
echo "  Model       : ${MODEL}"
echo "  TP size     : ${TP_SIZE}"
echo "  Tasks (${N_RESOLVED}): ${RESOLVED_TASKS}"
echo "  Max retries : ${MAX_RETRIES}"
echo "  Max steps   : ${MAX_STEPS}"
echo "  Output dir  : ${OUTPUT_DIR}"
echo "  HF cache    : ${HF_CACHE}"
echo "========================================================================"
echo ""

# ── Dry-run mode ──────────────────────────────────────────────────────────────

if [ "${DRY_RUN}" = "1" ]; then
    echo "[$(date +%H:%M:%S)] DRY RUN — skipping vLLM startup and collection"
    # shellcheck disable=SC2086
    python scripts/collect_safe_trajectories.py \
        --dry-run \
        --backend vllm \
        --vllm-url "${VLLM_URL}" \
        --model "${MODEL}" \
        --task-ids ${RESOLVED_TASKS}
    exit $?
fi

# ── Start vLLM server ─────────────────────────────────────────────────────────

echo "[$(date +%H:%M:%S)] Starting vLLM (tensor-parallel=${TP_SIZE})..."
python -m vllm.entrypoints.openai.api_server \
    --model "${MODEL}" \
    --tensor-parallel-size "${TP_SIZE}" \
    --port "${VLLM_PORT}" \
    --max-model-len 8192 \
    --dtype bfloat16 \
    --trust-remote-code \
    > "${LOG_DIR}/vllm_${SLURM_JOB_ID:-local}.log" 2>&1 &
VLLM_PID=$!
echo "  vLLM PID: ${VLLM_PID} — log: ${LOG_DIR}/vllm_${SLURM_JOB_ID:-local}.log"

# Ensure vLLM is killed when the job ends (timeout, scancel, error)
trap 'echo "[$(date +%H:%M:%S)] Caught signal — killing vLLM ${VLLM_PID}"; kill ${VLLM_PID} 2>/dev/null || true' EXIT

# ── Wait for vLLM health ──────────────────────────────────────────────────────

echo "[$(date +%H:%M:%S)] Waiting for vLLM /health ..."
MAX_WAIT=300
WAITED=0
until curl -sf "${VLLM_URL%/v1}/health" > /dev/null 2>&1; do
    sleep 5
    WAITED=$((WAITED + 5))
    if [ "${WAITED}" -ge "${MAX_WAIT}" ]; then
        echo "ERROR: vLLM did not come up after ${MAX_WAIT}s"
        exit 1
    fi
done
echo "[$(date +%H:%M:%S)] vLLM ready after ${WAITED}s"
echo ""

# ── Collect trajectories ──────────────────────────────────────────────────────

echo "[$(date +%H:%M:%S)] Starting collection (${N_RESOLVED} tasks, ${MAX_RETRIES} retries each)..."

# shellcheck disable=SC2086
python scripts/collect_safe_trajectories.py \
    --backend vllm \
    --model "${MODEL}" \
    --vllm-url "${VLLM_URL}" \
    --task-ids ${RESOLVED_TASKS} \
    --max-retries "${MAX_RETRIES}" \
    --max-steps "${MAX_STEPS}" \
    --output-dir "${OUTPUT_DIR}"

COLLECT_EXIT=$?

# ── Done ──────────────────────────────────────────────────────────────────────

echo ""
echo "[$(date +%H:%M:%S)] Collection finished (exit ${COLLECT_EXIT})"
echo "  Trajectories: ${OUTPUT_DIR}/"
ls -1 "${OUTPUT_DIR}"/*.json 2>/dev/null | wc -l | xargs printf "  JSON files saved: %s\n"

exit "${COLLECT_EXIT}"
