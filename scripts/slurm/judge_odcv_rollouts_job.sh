#!/bin/bash
# ABOUTME: SLURM wrapper for scripts/judge_odcv_rollouts.py — serves one open-weight judge model with vLLM on the job's GPUs and scores every arm in ROLLOUT_DIRS
# ABOUTME: Submit from the repo root: JUDGE_NAME=qwen3.6-27b JUDGE_MODEL=Qwen/Qwen3.6-27B ROLLOUT_DIRS="<arm dir> <arm dir>" sbatch --account=$ICRL_ACCOUNT --gres=gpu:l40s:2 --export=ALL scripts/slurm/judge_odcv_rollouts_job.sh
#
# One job per judge model; the script caches scores per judge and recomputes each
# arm's results.json from every judge cache present, so the second judge's job
# only adds its column. No hosted API is involved anywhere in this stage.
#SBATCH --job-name=odcv-judge
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=logs/slurm/%x_%j.out
#SBATCH --error=logs/slurm/%x_%j.err
set -euo pipefail
ICRL_REPO="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}"
cd "${ICRL_REPO}"
source scripts/slurm/job_environment.sh
export HF_HUB_OFFLINE=1
JUDGE_NAME="${JUDGE_NAME:?set JUDGE_NAME (cache name, e.g. qwen3.6-27b)}"
JUDGE_MODEL="${JUDGE_MODEL:?set JUDGE_MODEL (HF id or local dir)}"
ROLLOUT_DIRS="${ROLLOUT_DIRS:?set ROLLOUT_DIRS (space-separated arm directories)}"
TP="${TP:-$(nvidia-smi -L | wc -l)}"
PORT="${PORT:-8000}"
echo "=== odcv-judge === job ${SLURM_JOB_ID:-local} node $(hostname) commit $(git rev-parse --short HEAD) judge ${JUDGE_NAME} model ${JUDGE_MODEL} tp ${TP}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

# Model-specific serving flags: Qwen3.6 thinks (its reasoning is parsed out of the
# answer); Llama-3.3-70B needs nothing beyond tensor parallelism over four L40S.
# Mistral-Small-3.2 is served the way Mistral documents it: from consolidated
# weights with vLLM's own tokenizer/config/load formats. The HF-format path fails
# twice on this stack (job 5258505: transformers' MistralCommon tokenizer has no
# is_fast; job 5258546: no image processor for the vision tower).
case "${JUDGE_MODEL}" in
  *Qwen3.6*|*qwen3.6*) VLLM_MODEL_ARGS="${VLLM_MODEL_ARGS:---reasoning-parser qwen3}" ;;
  *Mistral*|*mistral*) VLLM_MODEL_ARGS="${VLLM_MODEL_ARGS:---tokenizer-mode mistral --config-format mistral --load-format mistral}" ;;
  *) VLLM_MODEL_ARGS="${VLLM_MODEL_ARGS:-}" ;;
esac
python -m vllm.entrypoints.openai.api_server --model "${JUDGE_MODEL}" --served-model-name judge \
  --tensor-parallel-size "${TP}" --port "${PORT}" --max-model-len "${MAX_MODEL_LEN:-65536}" \
  --max-num-seqs "${MAX_NUM_SEQS:-16}" --gpu-memory-utilization "${VLLM_GPU_FRACTION:-0.92}" \
  ${VLLM_MODEL_ARGS} > "logs/slurm/vllm_judge_${SLURM_JOB_ID:-local}.log" 2>&1 &
VLLM_PID=$!
trap 'kill ${VLLM_PID} 2>/dev/null || true' EXIT
for i in $(seq 1 240); do
  if curl -sf "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then echo "vLLM healthy after $((i*5))s"; break; fi
  kill -0 ${VLLM_PID} 2>/dev/null || { echo "vLLM died; tail of its log:"; tail -30 "logs/slurm/vllm_judge_${SLURM_JOB_ID:-local}.log"; exit 1; }
  sleep 5
done
curl -s "http://127.0.0.1:${PORT}/v1/models" | python -c "import json,sys; print('served:', [m['id'] for m in json.load(sys.stdin)['data']])"

ARGS=()
for d in ${ROLLOUT_DIRS}; do ARGS+=(--rollouts "${d}"); done
python scripts/judge_odcv_rollouts.py "${ARGS[@]}" --judge "${JUDGE_NAME}" --model judge \
  --base-url "http://127.0.0.1:${PORT}/v1" --workers "${JUDGE_WORKERS:-16}" ${JUDGES:+--judges ${JUDGES}} ${EXTRA:-}
echo "=== DONE ==="
