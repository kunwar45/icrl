#!/bin/bash
# ABOUTME: Runs the constraint-scorer smoke test on one L40S before a long fine-tuning chain
# ABOUTME: sbatch --account=aip-s2ganapa --gres=gpu:l40s:1 scratch/smoke_scenario_context_constraint_scorer_job.sh
#SBATCH --job-name=odcv-constraint-smoke
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=logs/slurm/%x_%j.out
#SBATCH --error=logs/slurm/%x_%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
source scripts/slurm/job_environment.sh
export HF_HUB_OFFLINE=1
python scratch/smoke_scenario_context_constraint_scorer.py \
  --model-dir "${SCRATCH}/constraint_models/odcv_context_rollouts" \
  --eval-root "${SCRATCH}/trajectories/odcv/eval"
echo "=== SMOKE OK ==="
