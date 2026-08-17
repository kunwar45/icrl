#!/usr/bin/env bash
# ABOUTME: Submits an ICRL trajectory-generation pass on killarney with the project's real paths
# ABOUTME: Run on the login node: bash submit_generation.sh <CYCLES> [TASK_IDS_CSV] [GPU_SPEC]
set -euo pipefail

CYCLES_ARG="${1:-1}"
TASK_IDS="${2:-}"
GPU_SPEC="${3:-gpu:l40s:4}"

cd /project/aip-s2ganapa/kunwar/icrl

# job_environment.sh defaults these at $SCRATCH, which is wrong on this cluster:
# code, venv and caches all live under /project. sbatch exports the submitting
# environment, so setting them here is what reaches the job.
export ICRL_VENV=/project/aip-s2ganapa/kunwar/venvs/icrl_v4
export REPOS_ROOT=/project/aip-s2ganapa/kunwar
export STWEBAGENT_ROOT=/project/aip-s2ganapa/kunwar/ST-WebAgentBench
export BROWSERGYM_ROOT=/project/aip-s2ganapa/kunwar/BrowserGym
export HF_HOME=/project/aip-s2ganapa/kunwar/hf_cache
export PLAYWRIGHT_BROWSERS_PATH=/project/aip-s2ganapa/kunwar/playwright-browsers
export SCRATCH="${SCRATCH:-/scratch/$USER}"

set -a
# shellcheck disable=SC1091
source .env
set +a

mkdir -p logs/slurm "${SCRATCH}/locks"

export CYCLES="${CYCLES_ARG}"
export RESEED_BEFORE_RUN=1
export CONFIG=configs/trajectory_generation/stwebagentbench_expert.yaml

# Deferred validation: verified on klogin03 2026-08-15 to give identical
# verdicts and identical reward at 4.27x the step speed (6.28s -> 1.47s).
# See scratch/probe_deferred_validation.py.
OVERRIDES="episode.defer_validation=true"
if [ -n "${TASK_IDS}" ]; then
    OVERRIDES="${OVERRIDES} benchmark.task_ids=[${TASK_IDS}]"
fi
export OVERRIDES
echo "overrides: ${OVERRIDES}"

echo "submitting: CYCLES=${CYCLES} gres=${GPU_SPEC} config=${CONFIG}"
sbatch --account=aip-s2ganapa --gres="${GPU_SPEC}" \
       --time=11:59:00 \
       scripts/slurm/generate_trajectories_job.sh
