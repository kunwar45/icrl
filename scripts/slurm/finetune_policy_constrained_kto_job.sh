#!/bin/bash
# ABOUTME: SLURM wrapper for scripts/finetune_policy_constrained_kto.py — one round per job; chain rounds with --dependency=afterok
# ABOUTME: Submit from the repo root: sbatch --account=$ICRL_ACCOUNT --gres=gpu:l40s:4 --export=ALL scripts/slurm/finetune_policy_constrained_kto_job.sh
#
# Four rounds, chained:
#   prev=$(sbatch --parsable ... this_job.sh); for i in 1 2 3; do prev=$(sbatch --parsable --dependency=afterok:$prev ... this_job.sh); done
# The script reads the next round from <output.dir>/state.json, so every job is the same command.
#SBATCH --job-name=odcv-constrained-kto
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --time=05:00:00
#SBATCH --output=logs/slurm/%x_%j.out
#SBATCH --error=logs/slurm/%x_%j.err
set -euo pipefail
ICRL_REPO="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}"
cd "${ICRL_REPO}"
source scripts/slurm/job_environment.sh
module load apptainer/1.4.5 2>/dev/null || module load apptainer 2>/dev/null || true
# Each sandbox cell runs in its own IPC namespace (src/environments/odcv/cell_driver.sh),
# so the node's System V queue table no longer fills; the tmpdir still goes on
# node-local scratch that SLURM wipes.
export APPTAINER_TMPDIR="${SLURM_TMPDIR:-/tmp}/apptainer_tmp" APPTAINER_CACHEDIR="${SCRATCH}/apptainer_cache"
mkdir -p "${APPTAINER_TMPDIR}" "${APPTAINER_CACHEDIR}"
apptainer instance stop --all >/dev/null 2>&1 || true
export HF_HUB_OFFLINE=1
CONFIG="${CONFIG:-configs/lagrangian_finetuning/odcv_constrained_kto.yaml}"
echo "=== odcv-constrained-kto === job ${SLURM_JOB_ID:-local} node $(hostname) config ${CONFIG} commit $(git rev-parse --short HEAD)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
apptainer --version
python scripts/finetune_policy_constrained_kto.py --config "${CONFIG}" ${EXTRA:-}
echo "=== DONE ==="
