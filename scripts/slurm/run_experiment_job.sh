#!/bin/bash
# ABOUTME: SLURM wrapper for scripts/run_experiment.py — the full ICRL pipeline (preflight through plots) in one job
# ABOUTME: Submit via scripts/submit_experiment.sh; PROFILE/RUN_NAME/STAGES/EXTRA env vars parameterize the run
# ============================================================================
# run_experiment_job.sh — full ICRL pipeline in one job
# ============================================================================
#
# Runs preflight → splits → encode → train C_theta → held-out gate → CuP
# baseline → Lagrangian fine-tune → CuP tuned → plots, via
# scripts/run_experiment.py.
#
# When it finishes, everything you need to look at is one file:
#   $SCRATCH/icrl/logs/$RUN_NAME/plots/report.html
# (figures inlined; scp it off the cluster and open it). The individual PNGs
# and, with EXTRA="--pdf", vector PDFs sit alongside it.
#
# Demos must already exist (scripts/slurm/collect_trajectories_job.sh — see
# docs/trajectory-collection.md). The driver reads either data/demos/*.jsonl or the
# collection output $SCRATCH/trajectories/<benchmark>/{expert,unsafe}.
#
# For the `cluster` profile SuiteCRM must be reachable at $WA_SUITECRM, because
# fine-tuning and CuP evaluation both roll out real browser episodes. The
# preflight stage checks that (plus GPUs, task registration and demo quality)
# before anything expensive starts.
#
# ── Usage ───────────────────────────────────────────────────────────────────
#
#   # Full run against the real benchmark:
#   sbatch scripts/slurm/run_experiment_job.sh
#
#   # Plumbing check on a GPU node — no browser, no CRM (~10 min):
#   PROFILE=smoke sbatch --gres=gpu:h100:1 --time=00:30:00 scripts/slurm/run_experiment_job.sh
#
#   # Only part of the pipeline:
#   STAGES=constraint,gate sbatch scripts/slurm/run_experiment_job.sh
#
#   # Extra Hydra overrides:
#   EXTRA="--override finetune.ppo.steps=50" sbatch scripts/slurm/run_experiment_job.sh
#
# ── Environment variables ───────────────────────────────────────────────────
#
#   PROFILE       smoke | local | cluster      (default: cluster)
#   RUN_NAME      experiment name              (default: icrl_$PROFILE)
#   STAGES        comma-separated stage subset (default: all)
#   STRICT_GATE   1 to abort when the AUROC gate fails (default: continue)
#   SAFE_DEMOS    .jsonl or trace dir          (default: auto-detected)
#   UNSAFE_DEMOS  .jsonl or trace dir          (default: auto-detected)
#   EXTRA         extra args passed to run_experiment.py
#
# NOTE: submit with scripts/submit_experiment.sh, which supplies --account,
# --gpus-per-node, --partition and --time on the sbatch command line (where they
# override the directives below). #SBATCH lines are literal text and cannot read
# environment variables, so anything baked in here is wrong on every cluster but
# the one it was written for. No --account default is set on purpose: a wrong
# account fails at submit time, while a missing one prints Slurm's own error.
#SBATCH --job-name=icrl-experiment
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/slurm/%x_%j.out
#SBATCH --error=logs/slurm/%x_%j.err

set -euo pipefail
# Slurm copies the batch script into a spool directory before running it, so
# "$(dirname "$0")" points at /cm/local/.../spool/job<N>/ and not at the repo.
# SLURM_SUBMIT_DIR is the directory sbatch was invoked from — the repo root.
ICRL_REPO="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "${ICRL_REPO}"
source "${ICRL_REPO}/scripts/slurm/job_environment.sh"

PROFILE="${PROFILE:-cluster}"
RUN_NAME="${RUN_NAME:-icrl_${PROFILE}}"
SCRATCH="${SCRATCH:-/scratch/${USER}}"

# Keep model weights on fast scratch rather than NFS home.
export HF_HOME="${HF_HOME:-${SCRATCH}/hf_cache}"
mkdir -p "${HF_HOME}" logs/slurm

ARGS=(--profile "${PROFILE}" --run-name "${RUN_NAME}")
[ -n "${STAGES:-}" ] && ARGS+=(--stages "${STAGES}")
[ "${STRICT_GATE:-0}" = "1" ] && ARGS+=(--strict-gate)
[ -n "${SAFE_DEMOS:-}" ] && ARGS+=(--safe-demos "${SAFE_DEMOS}")
[ -n "${UNSAFE_DEMOS:-}" ] && ARGS+=(--unsafe-demos "${UNSAFE_DEMOS}")

echo "=== ICRL experiment ==="
echo "cluster   : ${CC_CLUSTER:-unknown}"
echo "profile   : ${PROFILE}"
echo "run_name  : ${RUN_NAME}"
echo "node      : $(hostname)"
echo "offline   : HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-0}"
echo "HF_HOME   : ${HF_HOME}"
echo "SuiteCRM  : ${WA_SUITECRM:-<unset — cluster profile will fail>}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true
echo

# `set -e` would abort before the summary below, so handle the status ourselves.
set +e
python scripts/run_experiment.py "${ARGS[@]}" ${EXTRA:-}
STATUS=$?
set -e

echo
echo "=== report ==="
REPORT="$(find "${SCRATCH}/icrl/logs/${RUN_NAME}/plots" "logs/${RUN_NAME}/plots" \
            -name report.html 2>/dev/null | head -1)"
if [ -n "${REPORT}" ]; then
    echo "  ${REPORT}"
    echo "  copy it off the cluster:  scp ${USER}@$(hostname):${PWD}/${REPORT} ."
else
    echo "  no report.html — check the plots stage above"
fi

exit ${STATUS}
