#!/bin/bash
# ============================================================================
# submit_experiment.sh — build the sbatch command for THIS cluster and submit
# ============================================================================
#
# #SBATCH directives are literal text: they cannot read environment variables,
# so a script with `--account=def-s2ganapa` baked in only ever works on one
# cluster with one allocation. This wrapper passes those flags on the sbatch
# command line, where they override the directives in scripts/slurm/run_experiment_job.sh.
#
# ── Usage ───────────────────────────────────────────────────────────────────
#
#   bash scripts/infra/cluster_probe.sh              # discover the values first
#   export ICRL_ACCOUNT=aip-s2ganapa
#   export ICRL_GPU=l40s:1
#
#   # No browser, no CRM — the safe first submission:
#   bash scripts/infra/submit_experiment.sh --stages preflight,splits,encode,constraint,gate
#
#   # Everything, once SuiteCRM is up:
#   bash scripts/infra/submit_experiment.sh
#
#   # See the command without submitting:
#   DRY_RUN=1 bash scripts/infra/submit_experiment.sh
#
# ── Variables ───────────────────────────────────────────────────────────────
#
#   ICRL_ACCOUNT   allocation to charge          (required)
#   ICRL_GPU       "<type>:<count>" or "<count>" (default: 1 — omit for CPU-only)
#   ICRL_PARTITION partition                     (default: cluster's default)
#   ICRL_TIME      wall clock                    (default: 12:00:00)
#   ICRL_MEM       memory                        (default: 64G)
#   ICRL_CPUS      cpus-per-task                 (default: 8)
#   PROFILE        smoke | local | cluster       (default: cluster)
#   RUN_NAME       experiment name               (default: icrl_$PROFILE)
#   Anything after `--` or unrecognised is forwarded to run_experiment.py.
# ============================================================================

set -euo pipefail
cd "$(dirname "$0")/.."

PROFILE="${PROFILE:-cluster}"
RUN_NAME="${RUN_NAME:-icrl_${PROFILE}}"
TIME="${ICRL_TIME:-12:00:00}"
MEM="${ICRL_MEM:-64G}"
CPUS="${ICRL_CPUS:-8}"
GPU="${ICRL_GPU:-1}"

if [ -z "${ICRL_ACCOUNT:-}" ]; then
    cat >&2 <<'EOF'
ERROR: ICRL_ACCOUNT is not set.

Find your allocation:
    bash scripts/infra/cluster_probe.sh        # lists the accounts you can charge
    sshare -U -u $USER                   # or ask slurm directly

Then:
    export ICRL_ACCOUNT=<aip-... or def-...>
EOF
    exit 2
fi

SCRATCH="${SCRATCH:-/scratch/${USER}}"
export HF_HOME="${HF_HOME:-${SCRATCH}/hf_cache}"

# Compute nodes on Alliance clusters have no route to the internet. Models must
# already be in HF_HOME (scripts/infra/prefetch_models.py) or the job dies mid-run.
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"

# ── Partition ─────────────────────────────────────────────────────────────────
# Some clusters (Killarney) bin GPU partitions by maximum walltime —
# gpubase_l40s_b1 is 3h, _b2 12h, _b3 24h … — and there is no default that fits
# every duration, so a job with a mismatched --time is simply rejected. Ask
# Slurm which partitions exist, keep the ones offering the requested GPU type,
# and take the shortest bin that still fits the walltime (shortest = least
# contended). Clusters with a single GPU partition fall through untouched.

slurm_time_to_minutes() {
    local t="$1" d=0 h=0 m=0 s=0
    case "$t" in
        infinite|UNLIMITED|"") echo 99999999; return ;;
    esac
    case "$t" in *-*) d="${t%%-*}"; t="${t#*-}" ;; esac
    local IFS=:
    # shellcheck disable=SC2086
    set -- $t
    case $# in
        3) h=$1; m=$2; s=$3 ;;
        2) m=$1; s=$2 ;;
        1) m=$1 ;;
    esac
    echo $(( 10#${d:-0} * 1440 + 10#${h:-0} * 60 + 10#${m:-0} + (10#${s:-0} > 0 ? 1 : 0) ))
}

pick_partition() {
    local gtype="${1%%:*}" want want_min best best_min limit gres part
    command -v sinfo >/dev/null 2>&1 || return 0
    want_min=$(slurm_time_to_minutes "${TIME}")
    best=""; best_min=99999999

    while IFS='|' read -r part limit gres; do
        part="${part%\*}"                       # sinfo marks the default with *
        [ -n "${part}" ] || continue
        case "${gres}" in *gpu:${gtype}*) ;; *) continue ;; esac
        # Interactive/debug partitions are reserved for salloc and reject or
        # heavily restrict batch work — they tie with the short batch bin on
        # walltime, so exclude them explicitly rather than by ordering luck.
        case "${part}" in
            *interac*|*interactive*|*debug*|*test*) continue ;;
        esac
        local pmin
        pmin=$(slurm_time_to_minutes "${limit}")
        if [ "${pmin}" -ge "${want_min}" ] && [ "${pmin}" -lt "${best_min}" ]; then
            best="${part}"; best_min="${pmin}"
        fi
    done < <(sinfo -h -o "%P|%l|%G" 2>/dev/null | sort -u)

    [ -n "${best}" ] && echo "${best}"
}

if [ -z "${ICRL_PARTITION:-}" ] && [ -n "${GPU}" ] && [ "${GPU}" != "0" ]; then
    AUTO_PART=$(pick_partition "${GPU}")
    if [ -n "${AUTO_PART}" ]; then
        ICRL_PARTITION="${AUTO_PART}"
        echo "auto-selected partition ${ICRL_PARTITION} for ${GPU} @ ${TIME}"
    fi
fi

SBATCH_ARGS=(
    --account="${ICRL_ACCOUNT}"
    --time="${TIME}"
    --mem="${MEM}"
    --cpus-per-task="${CPUS}"
    --job-name="${RUN_NAME}"
)
[ -n "${ICRL_PARTITION:-}" ] && SBATCH_ARGS+=(--partition="${ICRL_PARTITION}")

# "l40s:1" → --gpus-per-node=l40s:1 ; "2" → --gpus-per-node=2 ; "" or "0" → CPU only
if [ -n "${GPU}" ] && [ "${GPU}" != "0" ]; then
    SBATCH_ARGS+=(--gpus-per-node="${GPU}")
fi

# Forwarded to the job, then on to run_experiment.py.
export PROFILE RUN_NAME
export EXTRA="${EXTRA:-} $*"

mkdir -p logs/slurm

echo "cluster   : ${CC_CLUSTER:-$(hostname)}"
echo "account   : ${ICRL_ACCOUNT}"
echo "gpus      : ${GPU:-none}"
echo "partition : ${ICRL_PARTITION:-<default>}"
echo "time/mem  : ${TIME} / ${MEM}, ${CPUS} cpus"
echo "profile   : ${PROFILE}   run_name: ${RUN_NAME}"
echo "HF_HOME   : ${HF_HOME}  (offline=${HF_HUB_OFFLINE})"
echo "extra     : ${EXTRA}"
echo
echo "sbatch ${SBATCH_ARGS[*]} scripts/slurm/run_experiment_job.sh"

if [ "${DRY_RUN:-0}" = "1" ]; then
    echo
    echo "DRY_RUN=1 — not submitted."
    exit 0
fi

# --export=ALL so PROFILE / RUN_NAME / EXTRA / HF_* reach the job script.
JOB=$(sbatch --parsable --export=ALL "${SBATCH_ARGS[@]}" scripts/slurm/run_experiment_job.sh)
echo
echo "submitted: job ${JOB}"
echo "  squeue -u ${USER} -j ${JOB}"
echo "  tail -f logs/slurm/${RUN_NAME}_${JOB}.out    # %x_%j = job-name_jobid"
echo "  scancel ${JOB}"
