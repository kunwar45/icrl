#!/usr/bin/env bash
# Uses apptainer instance run (not start) with --writable-tmpfs on read-only SIFs.
#
# Intended for Alliance / Compute Canada clusters: start once on the login node
# so SLURM jobs can point WA_SUITECRM at it instead of booting CRM per job.
#
# One-time image pull:
#   module load apptainer/1.4.5
#   mkdir -p /scratch/$USER/apptainer/tmp
#   APPTAINER_TMPDIR=/scratch/$USER/apptainer/tmp \
#     apptainer pull /scratch/$USER/apptainer/mariadb.sif \
#     docker://bitnamilegacy/mariadb:11.4
#   APPTAINER_TMPDIR=/scratch/$USER/apptainer/tmp \
#     apptainer pull /scratch/$USER/apptainer/suitecrm.sif \
#     docker://bitnamilegacy/suitecrm:8
#
# Usage:
#   bash scripts/start_suitecrm_apptainer.sh          # start instances
#   bash scripts/start_suitecrm_apptainer.sh --wait   # start and block until HTTP 200
#   bash scripts/start_suitecrm_apptainer.sh --stop   # stop instances
#   bash scripts/start_suitecrm_apptainer.sh --status # list instances
set -euo pipefail

MARIADB_SIF="${MARIADB_SIF:-/scratch/${USER}/apptainer/mariadb.sif}"
SUITECRM_SIF="${SUITECRM_SIF:-/scratch/${USER}/apptainer/suitecrm.sif}"
SUITECRM_DATA="${SUITECRM_DATA:-/scratch/${USER}/suitecrm}"
MARIADB_INSTANCE="${MARIADB_INSTANCE:-mariadb}"
SUITECRM_INSTANCE="${SUITECRM_INSTANCE:-suitecrm}"

usage() {
    sed -n '2,21p' "$0" | sed 's/^# \{0,1\}//'
}

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

require_sifs() {
    for sif in "${MARIADB_SIF}" "${SUITECRM_SIF}"; do
        if [ ! -f "${sif}" ]; then
            echo "ERROR: missing ${sif}" >&2
            echo "Pull with apptainer pull ${sif} docker://bitnamilegacy/..." >&2
            exit 1
        fi
    done
}

stop_instances() {
    load_apptainer
    apptainer instance stop "${SUITECRM_INSTANCE}" 2>/dev/null || true
    apptainer instance stop "${MARIADB_INSTANCE}" 2>/dev/null || true
    echo "Stopped ${MARIADB_INSTANCE} and ${SUITECRM_INSTANCE} instances."
}

start_instances() {
    load_apptainer
    require_sifs
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

    echo "Starting SuiteCRM instance (${SUITECRM_SIF})..."
    apptainer instance run \
        --writable-tmpfs \
        --bind "${SUITECRM_DATA}/app:/bitnami/suitecrm" \
        --env SUITECRM_DATABASE_HOST=127.0.0.1 \
        --env SUITECRM_DATABASE_PORT_NUMBER=3306 \
        --env SUITECRM_DATABASE_USER=bn_suitecrm \
        --env SUITECRM_DATABASE_NAME=bitnami_suitecrm \
        --env SUITECRM_DATABASE_PASSWORD=bitnami123 \
        --env ALLOW_EMPTY_PASSWORD=yes \
        "${SUITECRM_SIF}" "${SUITECRM_INSTANCE}"

    apptainer instance list
    echo ""
    echo "SuiteCRM URL for compute nodes:"
    echo "  WA_SUITECRM=http://$(hostname):8080/public"
}

wait_for_http() {
    local url="http://localhost:8080"
    echo "Waiting for SuiteCRM at ${url} (first boot can take ~10 min)..."
    until curl -sf "${url}" > /dev/null 2>&1; do
        sleep 15
        echo "  $(date +%H:%M:%S) waiting..."
    done
    echo "SuiteCRM is up at http://$(hostname):8080"
}

case "${1:-}" in
    -h|--help)
        usage
        exit 0
        ;;
    --stop)
        stop_instances
        ;;
    --status)
        load_apptainer
        apptainer instance list
        ;;
    --wait)
        start_instances
        wait_for_http
        ;;
    "")
        start_instances
        ;;
    *)
        echo "Unknown option: $1" >&2
        usage
        exit 2
        ;;
esac
