#!/usr/bin/env bash
# ABOUTME: Submits the ICRL unsafe trajectory-collection pass on killarney with the project's real paths
# ABOUTME: Run on the login node: bash submit_collection.sh <CYCLES> [AFTER_JOBID] [SHARD]
set -euo pipefail

CYCLES_ARG="${1:-4}"
AFTER_JOB="${2:-}"
TASK_IDS="${3:-}"
SHARD="${4:-}"

cd /project/aip-s2ganapa/kunwar/icrl

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
export CONFIG=configs/trajectory_collection/stwebagentbench_unsafe.yaml
# Same verified 4.27x step speedup as the generation pass. The unsafe keep rule
# reads the trajectory, never the benchmark reward, so nothing here depends on
# per-step scores.
OVERRIDES="episode.defer_validation=true"
# The unsafe task list MUST match whatever the expert set actually covers. The
# constraint is learned by contrast, so if the two classes span different tasks
# the head separates them by task identity and its AUROC looks fine while
# C_theta has learned nothing about safety. The config says as much at the top
# of its benchmark block; this is where it gets enforced per run.
if [ -n "${TASK_IDS}" ]; then
    OVERRIDES="${OVERRIDES} benchmark.task_ids=[${TASK_IDS}]"
fi
export OVERRIDES

DEP_ARGS=()
if [ -n "${AFTER_JOB}" ]; then
    # A dependency, NOT a race. The SuiteCRM lock fast-fails by design, so
    # submitting both at once would make this job start, fail to take the lock,
    # and exit in seconds — draining the chain rather than waiting its turn.
    DEP_ARGS=(--dependency="afterany:${AFTER_JOB}")
    echo "will start after job ${AFTER_JOB}"
fi

if [ -n "${SHARD}" ]; then
    # A different SuiteCRM entirely: its own lock, so this can run ALONGSIDE a
    # generation pass instead of behind it.
    export ICRL_SUITECRM_SHARD="${SHARD}"
    echo "using SuiteCRM shard ${SHARD}"
fi

echo "submitting collection: CYCLES=${CYCLES} overrides=${OVERRIDES}"
sbatch --account=aip-s2ganapa --gres=gpu:l40s:1 \
       --time=05:59:00 \
       "${DEP_ARGS[@]}" \
       scripts/slurm/collect_trajectories_job.sh
