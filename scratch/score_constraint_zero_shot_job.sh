#!/bin/bash
# ABOUTME: Runs the zero-shot prompted-constraint reference on one L40S
# ABOUTME: sbatch --account=aip-s2ganapa --gres=gpu:l40s:1 scratch/score_constraint_zero_shot_job.sh
#SBATCH --job-name=odcv-constraint-zeroshot
#SBATCH --nodes=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=96G
#SBATCH --time=03:00:00
#SBATCH --output=logs/slurm/%x_%j.out
#SBATCH --error=logs/slurm/%x_%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
source scripts/slurm/job_environment.sh
export HF_HUB_OFFLINE=1
D="${SCRATCH}/trajectories/odcv"
python scratch/score_constraint_zero_shot.py \
  --eval-root "$D/eval" \
  --splits "$D/numina_control_audited/split/splits.json" \
  --model Qwen/Qwen2.5-7B-Instruct \
  --out output/constraint_zero_shot_qwen2.5-7b.json
echo "=== DONE zeroshot ==="
