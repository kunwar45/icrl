#!/usr/bin/env bash
# ABOUTME: Starts N independent SuiteCRM+MariaDB stacks so N collection/generation passes can run at once
# ABOUTME: Run on a login node: bash scripts/start_suitecrm_shards.sh --shards 4 [--stop|--status]
#
# WHY shards exist: throughput on this pipeline is capped by the web app, not by
# GPUs. A pass that proves task completion from the database needs that database
# to itself — it reseeds, then attributes every change to its own episodes — so
# scripts/slurm/*_job.sh serialise on one lock and only one pass ever runs. Give
# each pass its own SuiteCRM and the cap lifts: K shards, K concurrent passes,
# on the GPUs already allocated.
#
# Each shard is fully independent:
#   instances     mariadb_shard_<n>, suitecrm_shard_<n>
#   data          $SCRATCH/suitecrm_shards/data_<n>/{mariadb,app}
#   sandbox       $SCRATCH/suitecrm_shards/sandbox_<n>   (copy of the base one)
#   http port     19080 + n
#   mariadb port  3306 + n
#   env file      $SCRATCH/suitecrm_shards/shard_<n>.env
#
# A job joins one by number:
#   ICRL_SUITECRM_SHARD=2 CONFIG=... sbatch scripts/slurm/generate_trajectories_job.sh
# job_environment.sh sources that shard's env file (overriding .env) and locks
# per shard, so passes on different shards never exclude each other.
#
# COSTS, before you pick a number:
#   - Each shard needs its own WRITABLE sandbox, because two apptainer instances
#     cannot share one --writable directory. That is a full copy of the SuiteCRM
#     sandbox per shard — several GB each on $SCRATCH.
#   - These run on the LOGIN NODE, like the single instance does. Four SuiteCRM
#     stacks is already a lot of login-node Apache and MariaDB; be a good
#     citizen and keep it small. 2-4 is the useful range.
#   - Compute nodes cannot host them today: `apptainer instance start` there
#     fails with "failed to connect to dbus ... rootless cgroup manager"
#     (see logs/slurm/icrl-gen_45204272.err). Moving shards onto the job's own
#     node means avoiding `instance start` entirely.
#
# After a shard first comes up its SuiteCRM login wizard has never been
# completed, and an agent session lands on a blank #/users/Wizard page that
# bounces all routing back to itself. Run the wizard script ONCE per shard —
# the script prints the exact command per shard when it finishes.
#
# Usage:
#   bash scripts/start_suitecrm_shards.sh --shards 4     # build + start 1..4
#   bash scripts/start_suitecrm_shards.sh --status
#   bash scripts/start_suitecrm_shards.sh --stop         # stop all shards
#   SHARD_HTTP_PORT_BASE=19200 bash scripts/start_suitecrm_shards.sh --shards 2
set -euo pipefail

_SCRATCH="${SCRATCH:-/scratch/${USER}}"
SHARD_ROOT="${SHARD_ROOT:-${_SCRATCH}/suitecrm_shards}"
BASE_SANDBOX="${SUITECRM_SANDBOX:-${_SCRATCH}/apptainer/suitecrm_sandbox}"
SHARD_HTTP_PORT_BASE="${SHARD_HTTP_PORT_BASE:-19080}"
SHARD_DB_PORT_BASE="${SHARD_DB_PORT_BASE:-3306}"
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

SHARDS=""
ACTION="start"
while [ $# -gt 0 ]; do
    case "$1" in
        --shards) SHARDS="$2"; shift 2 ;;
        --stop)   ACTION="stop"; shift ;;
        --status) ACTION="status"; shift ;;
        -h|--help)
            sed -n '/^# Usage/,/^set -e/p' "$0" | grep '^#' | sed 's/^# \{0,1\}//'
            exit 0 ;;
        *) echo "Unknown option: $1" >&2; exit 2 ;;
    esac
done

_shard_env_file() { echo "${SHARD_ROOT}/shard_$1.env"; }

# Every shard-specific setting in one place, exported for the delegate script.
_export_shard_settings() {  # _export_shard_settings <n>
    local n="$1"
    export MARIADB_INSTANCE="mariadb_shard_${n}"
    export SUITECRM_INSTANCE="suitecrm_shard_${n}"
    export SUITECRM_DATA="${SHARD_ROOT}/data_${n}"
    export SUITECRM_SANDBOX="${SHARD_ROOT}/sandbox_${n}"
    export SUITECRM_HTTP_PORT=$((SHARD_HTTP_PORT_BASE + n))
    export MARIADB_PORT=$((SHARD_DB_PORT_BASE + n))
    export ICRL_WA_ENV_FILE="$(_shard_env_file "${n}")"
}

_known_shards() {
    ls "${SHARD_ROOT}"/shard_*.env 2>/dev/null \
        | sed -E 's|.*/shard_([0-9]+)\.env|\1|' | sort -n
}

case "${ACTION}" in
  status)
    HOME="${_SCRATCH}" apptainer instance list 2>/dev/null || true
    echo ""
    for n in $(_known_shards); do
        echo "shard ${n}: $(cat "$(_shard_env_file "${n}")" | tr '\n' ' ')"
    done
    ;;

  stop)
    for n in $(_known_shards); do
        echo "Stopping shard ${n}..."
        _export_shard_settings "${n}"
        bash "${REPO_ROOT}/scripts/start_suitecrm_apptainer.sh" --stop || true
    done
    echo "All shards stopped. Data and sandboxes are left in ${SHARD_ROOT}."
    ;;

  start)
    [ -n "${SHARDS}" ] || { echo "ERROR: --shards N is required" >&2; exit 2; }
    case "${SHARDS}" in ''|*[!0-9]*) echo "ERROR: --shards needs a number" >&2; exit 2 ;; esac
    [ -d "${BASE_SANDBOX}" ] || {
        echo "ERROR: no base sandbox at ${BASE_SANDBOX}" >&2
        echo "       Build it once: apptainer build --sandbox ${BASE_SANDBOX} docker://bitnami/suitecrm:8" >&2
        exit 1
    }
    mkdir -p "${SHARD_ROOT}"

    for n in $(seq 1 "${SHARDS}"); do
        echo ""
        echo "═══ shard ${n}/${SHARDS} ═══"
        _export_shard_settings "${n}"

        # Copying rather than rebuilding from the SIF: a rebuild is minutes of
        # extraction per shard, a copy is a filesystem operation on the same
        # /scratch. Skipped when the shard sandbox already exists, so re-running
        # this script restarts shards without re-copying gigabytes.
        if [ ! -d "${SUITECRM_SANDBOX}" ]; then
            echo "Copying base sandbox → ${SUITECRM_SANDBOX} (several GB)..."
            cp -a "${BASE_SANDBOX}" "${SUITECRM_SANDBOX}"
        fi

        bash "${REPO_ROOT}/scripts/start_suitecrm_apptainer.sh"
    done

    echo ""
    echo "═══ ${SHARDS} shards up ═══"
    for n in $(seq 1 "${SHARDS}"); do
        echo "  shard ${n}: $(grep '^WA_SUITECRM=' "$(_shard_env_file "${n}")" | cut -d= -f2-)"
    done
    echo ""
    echo "Complete the login wizard ONCE per shard (an agent otherwise starts"
    echo "trapped on a blank #/users/Wizard page):"
    for n in $(seq 1 "${SHARDS}"); do
        echo "  set -a; . $(_shard_env_file "${n}"); set +a; PYTHONPATH=. python scratch/complete_wizard_via_ui.py"
    done
    echo ""
    echo "Then run passes against them, one per shard:"
    echo "  ICRL_SUITECRM_SHARD=1 CONFIG=... sbatch scripts/slurm/generate_trajectories_job.sh"
    ;;
esac
