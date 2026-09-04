#!/bin/bash
# ABOUTME: SLURM wrapper for scripts/collect_odcv_rollouts.py — serves the policy with vLLM on the job's GPUs, then rolls out ODCV cells through Apptainer sandboxes
# ABOUTME: Submit from the repo root: sbatch --account=$ICRL_ACCOUNT --gres=gpu:l40s:2 --export=ALL scripts/slurm/collect_odcv_rollouts_job.sh   (SMOKE=1 for the 2-scenario check)
#SBATCH --job-name=odcv-rollouts
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=logs/slurm/%x_%j.out
#SBATCH --error=logs/slurm/%x_%j.err
set -euo pipefail
ICRL_REPO="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}"
cd "${ICRL_REPO}"
source scripts/slurm/job_environment.sh
export HF_HUB_OFFLINE=1
CONFIG="${CONFIG:-configs/trajectory_collection/odcv_rollouts.yaml}"
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen3.6-27B}"
LORA_REPO="${LORA_REPO:-matboz/qwen3.6-27b-lora-9284-numina-control-716-r64}"
LORA_NAME="${ODCV_MODEL_NAME:-numina-control}"
TP="${TP:-$(nvidia-smi -L | wc -l)}"
PORT="${PORT:-8000}"
export ODCV_MODEL_NAME="${LORA_NAME}" ODCV_MODEL_URL="http://127.0.0.1:${PORT}/v1"
LORA_DIR="$(ls -d ${HF_HOME}/hub/models--${LORA_REPO//\//--}/snapshots/* | head -1)"
echo "=== odcv-rollouts === job ${SLURM_JOB_ID:-local} node $(hostname) commit $(git rev-parse --short HEAD) tp ${TP} lora ${LORA_DIR}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

# vLLM on the job's GPUs. Qwen3.6 needs the qwen3 reasoning parser, the qwen3_xml tool
# parser and a long context (LASR lost 7/80 cells at 40960); max_num_seqs bounds the
# hybrid-attention cache. The sandboxes run beside it on the CPUs.
python -m vllm.entrypoints.openai.api_server --model "${BASE_MODEL}" --served-model-name base \
  --enable-lora --lora-modules "${LORA_NAME}=${LORA_DIR}" --max-lora-rank 64 \
  --tensor-parallel-size "${TP}" --port "${PORT}" --max-model-len "${MAX_MODEL_LEN:-65536}" \
  --max-num-seqs "${MAX_NUM_SEQS:-16}" --gpu-memory-utilization "${VLLM_GPU_FRACTION:-0.90}" \
  --reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_xml \
  > "logs/slurm/vllm_${SLURM_JOB_ID:-local}.log" 2>&1 &
VLLM_PID=$!
trap 'kill ${VLLM_PID} 2>/dev/null || true' EXIT
for i in $(seq 1 240); do
  if curl -sf "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then echo "vLLM healthy after $((i*5))s"; break; fi
  kill -0 ${VLLM_PID} 2>/dev/null || { echo "vLLM died; tail of its log:"; tail -30 "logs/slurm/vllm_${SLURM_JOB_ID:-local}.log"; exit 1; }
  sleep 5
done
curl -s "http://127.0.0.1:${PORT}/v1/models" | python -c "import json,sys; print('served:', [m['id'] for m in json.load(sys.stdin)['data']])"

module load apptainer/1.4.5 2>/dev/null || module load apptainer 2>/dev/null || true
apptainer --version
EXTRA_ARGS=()
if [ "${SMOKE:-0}" = "1" ]; then
  EXTRA_ARGS+=(--override "benchmark.scenarios=[Ai-Pharma-Quality-Control-Data-Suppression,Academic-Research-Integrity-Finding]"
               --override "rollouts.concurrency=4" --override "output.dir=${SCRATCH}/trajectories/odcv/rollouts/smoke_${SLURM_JOB_ID:-local}")
fi
python scripts/collect_odcv_rollouts.py --config "${CONFIG}" "${EXTRA_ARGS[@]}" ${EXTRA:-}
echo "=== DONE ==="
