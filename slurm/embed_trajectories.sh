#!/bin/bash
# ============================================================================
# embed_trajectories.sh — Pre-compute trajectory embeddings (frozen Qwen-1.5B)
# ============================================================================
#
# Loads the frozen Qwen/Qwen2.5-1.5B backbone, mean-pools last hidden states
# over each trajectory, and saves .pt bundles for the ICRL constraint trainer.
#
# Run this AFTER gen_safe_demos.sh has produced trajectory JSON files.
# The resulting .pt files are loaded directly by ICRLTrainer — the backbone
# never needs to run again during training.
#
# ── Quick start ──────────────────────────────────────────────────────────────
#
#   # Encode safe trajectories from default dir:
#   sbatch slurm/embed_trajectories.sh
#
#   # Custom input / output paths:
#   SAFE_DIR=/scratch/$USER/traj/safe sbatch slurm/embed_trajectories.sh
#
#   # Larger batch (more VRAM) + flash attention (A100/H100):
#   BATCH=32 FLASH_ATTN=1 sbatch slurm/embed_trajectories.sh
#
#   # Dry-run: loads model and encodes one dummy text, then exits:
#   DRY_RUN=1 sbatch slurm/embed_trajectories.sh
#
# ── Environment variables ────────────────────────────────────────────────────
#
#   SAFE_DIR       Directory of task_*_trace_*.json files (new format).
#                  Default: trajectories/safe (relative to repo root).
#
#   UNSAFE_JSONL   Path to unsafe JSONL file (old format).
#                  Default: data/demos/stwebagent_unsafe.jsonl
#
#   UNSAFE_DIR     Directory of unsafe task_*_trace_*.json files (new format).
#                  Checked only if UNSAFE_JSONL does not exist.
#
#   SAFE_OUT       Output path for safe embeddings .pt file.
#                  Default: embeddings/safe.pt
#
#   UNSAFE_OUT     Output path for unsafe embeddings .pt file.
#                  Default: embeddings/unsafe.pt
#
#   MODEL          Encoder model ID (default: Qwen/Qwen2.5-1.5B).
#                  Keep the base (non-Instruct) model for best mean-pool quality.
#
#   MAX_LEN        Token limit per trajectory (default: 2048).
#
#   BATCH          Encoding batch size (default: 16; raise for >40 GB VRAM).
#
#   FLASH_ATTN     Set to 1 to enable flash_attention_2 (A100/H100 only).
#                  Saves ~40% VRAM for long sequences.
#
#   DRY_RUN        Set to 1 to load model, encode a dummy text, and exit.
#
# ── Resource sizing ──────────────────────────────────────────────────────────
#
#   Qwen2.5-1.5B in bfloat16 ≈ 3 GB VRAM → comfortably fits one A100-40GB.
#   100 trajectories × 2048 tokens ≈ 2-5 min with batch_size=16.
#
#SBATCH --job-name=icrl-embed
#SBATCH --account=def-s2ganapa
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=logs/slurm/%x_%j.out
#SBATCH --error=logs/slurm/%x_%j.err

set -euo pipefail

# ── Defaults (override via env vars before sbatch) ───────────────────────────

MODEL="${MODEL:-Qwen/Qwen2.5-1.5B}"
MAX_LEN="${MAX_LEN:-2048}"
BATCH="${BATCH:-16}"
FLASH_ATTN="${FLASH_ATTN:-0}"
DRY_RUN="${DRY_RUN:-0}"

SAFE_DIR="${SAFE_DIR:-trajectories/safe}"
UNSAFE_JSONL="${UNSAFE_JSONL:-data/demos/stwebagent_unsafe.jsonl}"
UNSAFE_DIR="${UNSAFE_DIR:-trajectories/unsafe}"

SAFE_OUT="${SAFE_OUT:-embeddings/safe.pt}"
UNSAFE_OUT="${UNSAFE_OUT:-embeddings/unsafe.pt}"

HF_CACHE="${SCRATCH:-/tmp}/hf_cache"
LOG_DIR="logs/slurm"

# ── Setup ─────────────────────────────────────────────────────────────────────

mkdir -p "${LOG_DIR}" embeddings "${HF_CACHE}"

export HF_HOME="${HF_CACHE}"
export TRANSFORMERS_CACHE="${HF_CACHE}"

source "$(dirname "$0")/env.sh" 2>/dev/null || true
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

FLASH_FLAG=""
[ "${FLASH_ATTN}" = "1" ] && FLASH_FLAG="--flash-attn"

DRY_FLAG=""
[ "${DRY_RUN}" = "1" ] && DRY_FLAG="--dry-run"

echo "========================================================================"
echo " icrl-embed  job ${SLURM_JOB_ID:-local}  node $(hostname)"
echo "========================================================================"
echo "  Model      : ${MODEL}"
echo "  Max length : ${MAX_LEN}"
echo "  Batch size : ${BATCH}"
echo "  Flash attn : ${FLASH_ATTN}"
echo "  Safe dir   : ${SAFE_DIR}"
echo "  Unsafe src : ${UNSAFE_JSONL} (or ${UNSAFE_DIR})"
echo "  Safe out   : ${SAFE_OUT}"
echo "  Unsafe out : ${UNSAFE_OUT}"
echo "  HF cache   : ${HF_CACHE}"
echo "========================================================================"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true
echo ""

# ── Encode safe trajectories ──────────────────────────────────────────────────

if [ -d "${SAFE_DIR}" ]; then
    N_SAFE=$(find "${SAFE_DIR}" -name "task_*_trace_*.json" | wc -l | tr -d ' ')
    echo "[$(date +%H:%M:%S)] Encoding ${N_SAFE} SAFE trajectories from ${SAFE_DIR} ..."
    python scripts/encode_trajectories.py \
        --input-dir "${SAFE_DIR}" \
        --label safe \
        --output "${SAFE_OUT}" \
        --model "${MODEL}" \
        --max-length "${MAX_LEN}" \
        --batch-size "${BATCH}" \
        ${FLASH_FLAG} ${DRY_FLAG}
    echo "[$(date +%H:%M:%S)] Safe embeddings → ${SAFE_OUT}"
else
    echo "[WARN] ${SAFE_DIR} not found — skipping safe encoding."
    echo "       Run gen_safe_demos.sh first."
fi

# ── Encode unsafe trajectories ────────────────────────────────────────────────

echo ""
if [ -f "${UNSAFE_JSONL}" ]; then
    N_UNSAFE=$(wc -l < "${UNSAFE_JSONL}" | tr -d ' ')
    echo "[$(date +%H:%M:%S)] Encoding ${N_UNSAFE} UNSAFE trajectories from ${UNSAFE_JSONL} ..."
    python scripts/encode_trajectories.py \
        --jsonl "${UNSAFE_JSONL}" \
        --label unsafe \
        --output "${UNSAFE_OUT}" \
        --model "${MODEL}" \
        --max-length "${MAX_LEN}" \
        --batch-size "${BATCH}" \
        ${FLASH_FLAG} ${DRY_FLAG}
    echo "[$(date +%H:%M:%S)] Unsafe embeddings → ${UNSAFE_OUT}"
elif [ -d "${UNSAFE_DIR}" ]; then
    N_UNSAFE=$(find "${UNSAFE_DIR}" -name "task_*_trace_*.json" | wc -l | tr -d ' ')
    echo "[$(date +%H:%M:%S)] Encoding ${N_UNSAFE} UNSAFE trajectories from ${UNSAFE_DIR} ..."
    python scripts/encode_trajectories.py \
        --input-dir "${UNSAFE_DIR}" \
        --label unsafe \
        --output "${UNSAFE_OUT}" \
        --model "${MODEL}" \
        --max-length "${MAX_LEN}" \
        --batch-size "${BATCH}" \
        ${FLASH_FLAG} ${DRY_FLAG}
    echo "[$(date +%H:%M:%S)] Unsafe embeddings → ${UNSAFE_OUT}"
else
    echo "[INFO] No unsafe trajectory source found — skipping."
    echo "       Set UNSAFE_JSONL or UNSAFE_DIR if you have unsafe demos."
fi

# ── Summary ───────────────────────────────────────────────────────────────────

echo ""
echo "[$(date +%H:%M:%S)] Done."
ls -lh embeddings/*.pt 2>/dev/null || true
