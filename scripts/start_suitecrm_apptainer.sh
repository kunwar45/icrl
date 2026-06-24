#!/usr/bin/env bash
# Starts MariaDB + SuiteCRM as Apptainer instances on the login node.
#
# Intended for Alliance / Compute Canada clusters: start once on the login node
# so SLURM jobs can point WA_SUITECRM at it instead of booting CRM per job.
#
# WHY sandbox + --writable for SuiteCRM:
#   The cluster's apptainer.conf sets AllowSetuidMountExtfs=false, so the
#   --overlay ext3-image approach fails with "permission denied" on this system.
#   --writable-tmpfs fills the small default tmpfs (64 MB) and hangs boot.
#   Solution: extract the SIF to a sandbox directory on /scratch once, then
#   run the sandbox with --writable. Writes go directly to /scratch (no size
#   limits) and the .angular build cache can be created and removed freely.
#
# One-time setup:
#   module load apptainer/1.4.5
#   mkdir -p /scratch/$USER/apptainer/tmp
#   export APPTAINER_TMPDIR=/scratch/$USER/apptainer/tmp
#   apptainer pull /scratch/$USER/apptainer/mariadb.sif docker://bitnamilegacy/mariadb:11.4
#   apptainer pull /scratch/$USER/apptainer/suitecrm.sif docker://bitnamilegacy/suitecrm:8
#   # Build the writable sandbox (only needed once):
#   apptainer build --sandbox /scratch/$USER/apptainer/suitecrm_sandbox \
#       /scratch/$USER/apptainer/suitecrm.sif
#
# Usage:
#   bash scripts/start_suitecrm_apptainer.sh           # start + wait for HTTP
#   bash scripts/start_suitecrm_apptainer.sh --stop    # stop instances
#   bash scripts/start_suitecrm_apptainer.sh --status  # list instances
#   bash scripts/start_suitecrm_apptainer.sh --rebuild-sandbox  # re-extract SIF → sandbox
set -euo pipefail

MARIADB_SIF="${MARIADB_SIF:-/scratch/${USER}/apptainer/mariadb.sif}"
SUITECRM_SIF="${SUITECRM_SIF:-/scratch/${USER}/apptainer/suitecrm.sif}"
SUITECRM_SANDBOX="${SUITECRM_SANDBOX:-/scratch/${USER}/apptainer/suitecrm_sandbox}"
SUITECRM_DATA="${SUITECRM_DATA:-/scratch/${USER}/suitecrm}"
MARIADB_INSTANCE="${MARIADB_INSTANCE:-mariadb}"
SUITECRM_INSTANCE="${SUITECRM_INSTANCE:-suitecrm}"

load_apptainer() {
    if command -v apptainer &>/dev/null; then
        return 0
    fi
    for mod in apptainer/1.4.5 apptainer/1.3.5; do
        if module load "$mod" 2>/dev/null && command -v apptainer &>/dev/null; then
            echo "Loaded ${mod}"
            return 0
        fi
    done
    echo "ERROR: apptainer not found. Try: module load apptainer/1.4.5" >&2
    exit 127
}

require_images() {
    if [ ! -f "${MARIADB_SIF}" ]; then
        echo "ERROR: missing ${MARIADB_SIF}" >&2
        echo "Pull with: apptainer pull ${MARIADB_SIF} docker://bitnamilegacy/mariadb:11.4" >&2
        exit 1
    fi
    if [ ! -d "${SUITECRM_SANDBOX}" ]; then
        echo "ERROR: missing sandbox ${SUITECRM_SANDBOX}" >&2
        echo "Build with: apptainer build --sandbox ${SUITECRM_SANDBOX} ${SUITECRM_SIF}" >&2
        exit 1
    fi
}

build_sandbox() {
    if [ ! -f "${SUITECRM_SIF}" ]; then
        echo "ERROR: missing ${SUITECRM_SIF} — pull first:" >&2
        echo "  apptainer pull ${SUITECRM_SIF} docker://bitnamilegacy/suitecrm:8" >&2
        exit 1
    fi
    echo "Building writable sandbox from ${SUITECRM_SIF}..."
    mkdir -p "$(dirname "${SUITECRM_SANDBOX}")"
    apptainer build --sandbox "${SUITECRM_SANDBOX}" "${SUITECRM_SIF}"
    echo "Sandbox ready at ${SUITECRM_SANDBOX}"
}

stop_instances() {
    load_apptainer
    apptainer instance stop "${SUITECRM_INSTANCE}" 2>/dev/null || true
    apptainer instance stop "${MARIADB_INSTANCE}" 2>/dev/null || true
    echo "Stopped ${MARIADB_INSTANCE} and ${SUITECRM_INSTANCE} instances."
}

start_instances() {
    load_apptainer
    require_images
    mkdir -p "${SUITECRM_DATA}/mariadb" "${SUITECRM_DATA}/app"

    apptainer instance stop "${SUITECRM_INSTANCE}" 2>/dev/null || true
    apptainer instance stop "${MARIADB_INSTANCE}" 2>/dev/null || true

    echo "Starting MariaDB instance (${MARIADB_SIF})..."
    apptainer instance run \
        --writable-tmpfs \
        --bind "${SUITECRM_DATA}/mariadb:/bitnami/mariadb" \
        --env ALLOW_EMPTY_PASSWORD=yes \
        --env MARIADB_USER=bn_suitecrm \
        --env MARIADB_DATABASE=bitnami_suitecrm \
        --env MARIADB_PASSWORD=bitnami123 \
        "${MARIADB_SIF}" "${MARIADB_INSTANCE}"

    echo "Waiting 30 s for MariaDB..."
    sleep 30

    # Use --writable sandbox: AllowSetuidMountExtfs=false on this cluster means
    # --overlay ext3 fails; --writable-tmpfs fills the default 64 MB tmpfs.
    # The sandbox on /scratch has no space constraints.
    echo "Starting SuiteCRM instance (sandbox: ${SUITECRM_SANDBOX})..."
    apptainer instance run \
        --writable \
        --bind "${SUITECRM_DATA}/app:/bitnami/suitecrm" \
        --env SUITECRM_DATABASE_HOST=127.0.0.1 \
        --env SUITECRM_DATABASE_PORT_NUMBER=3306 \
        --env SUITECRM_DATABASE_USER=bn_suitecrm \
        --env SUITECRM_DATABASE_NAME=bitnami_suitecrm \
        --env SUITECRM_DATABASE_PASSWORD=bitnami123 \
        --env ALLOW_EMPTY_PASSWORD=yes \
        "${SUITECRM_SANDBOX}" "${SUITECRM_INSTANCE}"

    apptainer instance list
    echo ""
    WA_SUITECRM_URL="http://$(hostname):8080/public"
    echo "SuiteCRM URL for compute nodes:"
    echo "  WA_SUITECRM=${WA_SUITECRM_URL}"
    # Persist to scratch so SLURM jobs can pick it up even if /home Lustre is degraded
    printf 'WA_SUITECRM=%s\n' "${WA_SUITECRM_URL}" > "/scratch/${USER}/icrl_wa_env"
    echo "  → saved to /scratch/${USER}/icrl_wa_env"
}

wait_for_http() {
    local url="http://localhost:8080"
    local max_wait=900
    local waited=0
    echo "Waiting for SuiteCRM at ${url} (first boot ~10-15 min, subsequent ~1 min)..."
    until curl -sf "${url}" > /dev/null 2>&1; do
        sleep 15
        waited=$((waited + 15))
        echo "  $(date +%H:%M:%S) waiting... (${waited}s / ${max_wait}s)"
        if [ "${waited}" -ge "${max_wait}" ]; then
            echo "ERROR: SuiteCRM did not respond after ${max_wait}s." >&2
            echo "Check logs: tail ~/.apptainer/instances/logs/\$(hostname)/\$USER/suitecrm.err" >&2
            echo "To rebuild sandbox: bash scripts/start_suitecrm_apptainer.sh --rebuild-sandbox" >&2
            exit 1
        fi
    done
    echo "SuiteCRM is up at http://$(hostname):8080"
}

case "${1:-}" in
    -h|--help)
        sed -n '/^# Usage/,/^[^#]/p' "$0" | grep '^#' | sed 's/^# \{0,1\}//'
        exit 0
        ;;
    --stop)
        stop_instances
        ;;
    --status)
        load_apptainer
        apptainer instance list
        ;;
    --rebuild-sandbox)
        echo "Deleting and rebuilding SuiteCRM sandbox from SIF..."
        rm -rf "${SUITECRM_SANDBOX}"
        build_sandbox
        echo "Done. Run 'bash scripts/start_suitecrm_apptainer.sh' to start."
        ;;
    "")
        start_instances
        wait_for_http
        ;;
    *)
        echo "Unknown option: $1" >&2
        exit 2
        ;;
esac
