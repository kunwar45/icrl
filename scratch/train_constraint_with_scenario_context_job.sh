#!/bin/bash
# ABOUTME: Trains one arm of the scenario-context constraint ablation on a single L40S
# ABOUTME: sbatch --account=aip-s2ganapa --gres=gpu:l40s:1 --export=ALL,ARM=<arm> scratch/train_constraint_with_scenario_context_job.sh
#SBATCH --job-name=odcv-constraint-context
#SBATCH --nodes=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --output=logs/slurm/%x_%j.out
#SBATCH --error=logs/slurm/%x_%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
source scripts/slurm/job_environment.sh
export HF_HUB_OFFLINE=1
D="${SCRATCH}/trajectories/odcv"
echo "ARM=${ARM} TRAIN_CAP=${TRAIN_CAP:-0} SAVE_MODEL=${SAVE_MODEL:-}"
python scratch/train_constraint_with_scenario_context.py \
  --expert-dir "$D/numina_control_audited/expert" \
  --unsafe-dir "$D/numina_control_audited/unsafe" \
  --eval-root "$D/eval" \
  --splits "$D/numina_control_audited/split/splits.json" \
  --arm "${ARM}" \
  --encoder Qwen/Qwen2.5-7B-Instruct \
  --max-length 3072 --batch 1 --grad-accum 8 --eval-batch 2 \
  --epochs 6 --lr 5e-5 --lora-r 16 --train-cap "${TRAIN_CAP:-0}" --seed "${SEED:-0}" \
  --out "output/constraint_context_${ARM}${OUT_SUFFIX:-}.json" \
  ${SAVE_MODEL:+--save-model "${SCRATCH}/constraint_models/odcv_${ARM}"}
echo "=== DONE ${ARM} ==="
