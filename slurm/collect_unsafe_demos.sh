#!/bin/bash
# Collect unsafe (adversarial) demonstrations from ST-WebAgentBench using Qwen-2.5-7B.
#
# Strategy: run WITHOUT safety system prompt. Smaller model naturally skips
# confirmation steps and violates policies — exactly the adversarial signal ICRL needs.
# The agent doesn't need to succeed at the task; policy violations are what matter.
#
# Usage:
#   sbatch slurm/collect_unsafe_demos.sh
#   sbatch slurm/collect_unsafe_demos.sh --task-limit 50   # smoke test
#
# Runs on 1× A100 (Qwen-7B fits comfortably).

#SBATCH --job-name=icrl-unsafe-demos
#SBATCH --account=def-s2ganapa
#SBATCH --nodes=1
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=logs/slurm/%x_%j.out
#SBATCH --error=logs/slurm/%x_%j.err

set -euo pipefail

source "$(dirname "$0")/env.sh"
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

mkdir -p logs/slurm data/demos

# ── Start vLLM server ─────────────────────────────────────────────────────────
echo "[$(date)] Starting vLLM server for Qwen-2.5-7B (tensor-parallel=1)..."
vllm serve Qwen/Qwen2.5-7B-Instruct \
    --tensor-parallel-size 1 \
    --port 8000 \
    --max-model-len 8192 \
    > logs/slurm/vllm_unsafe_${SLURM_JOB_ID}.log 2>&1 &
VLLM_PID=$!

echo "[$(date)] Waiting for vLLM server to be ready..."
until curl -sf http://localhost:8000/health > /dev/null 2>&1; do sleep 5; done
echo "[$(date)] vLLM server ready."

# ── Run collection ─────────────────────────────────────────────────────────────
python scripts/collect_stwebagent_demos.py \
    --mode unsafe \
    --model Qwen/Qwen2.5-7B-Instruct \
    --vllm-base-url http://localhost:8000/v1 \
    --n-rollouts 5 \
    --max-steps 30 \
    --benchmark-root "${STWEBAGENT_ROOT}" \
    --output data/demos/stwebagent_unsafe.jsonl \
    "$@"

# ── Cleanup ───────────────────────────────────────────────────────────────────
echo "[$(date)] Shutting down vLLM server (PID $VLLM_PID)..."
kill $VLLM_PID 2>/dev/null || true
wait $VLLM_PID 2>/dev/null || true
echo "[$(date)] Done."
