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
#   bash scripts/start_suitecrm_apptainer.sh                # start + wait for HTTP
#   bash scripts/start_suitecrm_apptainer.sh --stop         # stop instances
#   bash scripts/start_suitecrm_apptainer.sh --status       # list instances
#   bash scripts/start_suitecrm_apptainer.sh --rebuild-sandbox  # re-extract SIF → sandbox
#   bash scripts/start_suitecrm_apptainer.sh --fresh-install    # wipe data + reinstall
#
# Port notes:
#   Bitnami's appinit always writes port 8080 to the live Apache conf/ directory,
#   regardless of APACHE_HTTP_PORT_NUMBER or pre-patched conf files.
#   This script works around that by exec-ing into the running instance after
#   appinit finishes, patching the conf files from inside, and restarting Apache.
set -euo pipefail

# /home Lustre is degraded on some clusters — redirect apptainer instance state
# to $SCRATCH so instance JSON files are written to a healthy filesystem.
_APTY_HOME="${SCRATCH:-/scratch/${USER}}"
mkdir -p "${_APTY_HOME}/.apptainer"
apptainer() { HOME="${_APTY_HOME}" command apptainer "$@"; }

_SCRATCH="${SCRATCH:-/scratch/${USER}}"
MARIADB_SIF="${MARIADB_SIF:-${_SCRATCH}/apptainer/mariadb.sif}"
SUITECRM_SIF="${SUITECRM_SIF:-${_SCRATCH}/apptainer/suitecrm.sif}"
SUITECRM_SANDBOX="${SUITECRM_SANDBOX:-${_SCRATCH}/apptainer/suitecrm_sandbox}"
SUITECRM_DATA="${SUITECRM_DATA:-${_SCRATCH}/suitecrm}"
MARIADB_INSTANCE="${MARIADB_INSTANCE:-mariadb}"
SUITECRM_INSTANCE="${SUITECRM_INSTANCE:-suitecrm}"

# Auto-select a free port if the caller didn't set one.
# Scans 19080-19099 (high range, unlikely to collide with other users).
_find_free_port() {
    local p
    for p in $(seq 19080 19099); do
        ss -tlnp 2>/dev/null | grep -qE ":${p}[[:space:]]|:${p}$" || { echo "${p}"; return 0; }
    done
    python3 -c "import socket; s=socket.socket(); s.bind(('',0)); print(s.getsockname()[1]); s.close()" 2>/dev/null || echo "19080"
}
if [ -z "${SUITECRM_HTTP_PORT:-}" ]; then
    SUITECRM_HTTP_PORT="$(_find_free_port)"
    echo "Auto-selected free port: ${SUITECRM_HTTP_PORT}"
fi

load_apptainer() {
    if command -v apptainer &>/dev/null; then
        return 0
    fi
    # Source lmod init if module command is not available (non-interactive bash)
    if ! command -v module &>/dev/null; then
        for _lmod_init in \
            /cvmfs/soft.computecanada.ca/nix/var/nix/profiles/16.09/lmod/lmod/init/bash \
            /cvmfs/soft.computecanada.ca/custom/software/lmod/lmod/init/profile \
            /etc/profile.d/lmod.sh \
            /usr/share/lmod/lmod/init/bash; do
            if [ -f "${_lmod_init}" ]; then
                # shellcheck disable=SC1090
                source "${_lmod_init}" 2>/dev/null && break
            fi
        done
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
    # Alliance clusters auto-bind /project and /scratch into every container.
    # With --writable, apptainer can't auto-create missing destinations — they
    # must exist inside the sandbox or startup fails.
    mkdir -p "${SUITECRM_SANDBOX}/project" "${SUITECRM_SANDBOX}/scratch"
}

build_sandbox() {
    if [ ! -f "${SUITECRM_SIF}" ]; then
        echo "ERROR: missing ${SUITECRM_SIF} — pull first:" >&2
        echo "  apptainer pull ${SUITECRM_SIF} docker://bitnamilegacy/suitecrm:8" >&2
        exit 1
    fi
    export APPTAINER_TMPDIR="${APPTAINER_TMPDIR:-${_SCRATCH}/apptainer/tmp}"
    mkdir -p "${APPTAINER_TMPDIR}" "$(dirname "${SUITECRM_SANDBOX}")"
    echo "Building writable sandbox from ${SUITECRM_SIF}..."
    echo "  APPTAINER_TMPDIR=${APPTAINER_TMPDIR}"
    apptainer build --sandbox "${SUITECRM_SANDBOX}" "${SUITECRM_SIF}"
    echo "Sandbox ready at ${SUITECRM_SANDBOX}"
}

stop_instances() {
    load_apptainer
    apptainer instance stop "${SUITECRM_INSTANCE}" 2>/dev/null || true
    apptainer instance stop "${MARIADB_INSTANCE}" 2>/dev/null || true
    echo "Stopped ${MARIADB_INSTANCE} and ${SUITECRM_INSTANCE} instances."
}

# Exec into the running SuiteCRM instance, patch Apache conf files to use
# SUITECRM_HTTP_PORT, and restart Apache.
#
# WHY exec-based instead of pre-patching host files:
#   Bitnami's appinit always runs "Configuring the HTTP port" which writes
#   port 8080 into the live conf/ directory, overriding any pre-patched files
#   and ignoring the APACHE_HTTP_PORT_NUMBER env var. The only reliable approach
#   is to patch AFTER appinit writes the files (from inside the container).
_exec_fix_apache_port() {
    local port="${SUITECRM_HTTP_PORT}"
    local next=$((port + 1))
    echo "  Exec-patching Apache config inside ${SUITECRM_INSTANCE} instance..."
    apptainer exec instance://"${SUITECRM_INSTANCE}" bash -s <<INNERSCRIPT
set -e
PATCHED=0
while IFS= read -r f; do
    if grep -qE '8080|8081' "\$f" 2>/dev/null; then
        sed -i \
            -e 's/Listen 8080/Listen ${port}/g' \
            -e 's/:8080>/:${port}>/g' \
            -e 's/Listen 8081/Listen ${next}/g' \
            -e 's/:8081>/:${next}>/g' \
            "\$f"
        echo "    patched: \$f"
        PATCHED=\$((PATCHED + 1))
    fi
done < <(find /opt/bitnami/apache -name '*.conf' -type f 2>/dev/null)
echo "  Patched \${PATCHED} file(s)."

HTTPD=\$(find /opt/bitnami -name 'httpd' -perm -u+x 2>/dev/null | grep -v '\.bak' | head -1)
if [ -z "\$HTTPD" ]; then
    echo "ERROR: httpd binary not found inside container" >&2
    find /opt/bitnami -name 'httpd' 2>/dev/null >&2
    exit 1
fi
CONF=\$(find /opt/bitnami/apache/conf -maxdepth 1 -name 'httpd.conf' 2>/dev/null | head -1)
echo "  Stopping old Apache (\$HTTPD)..."
"\$HTTPD" -k stop 2>/dev/null || killall httpd 2>/dev/null || true
sleep 3
echo "  Starting Apache on port ${port}..."
"\$HTTPD" -f "\$CONF" 2>&1 &
disown
echo "  Apache restarted (port ${port})"
INNERSCRIPT
}

start_instances() {
    load_apptainer
    require_images
    mkdir -p "${SUITECRM_DATA}/mariadb" "${SUITECRM_DATA}/app"

    apptainer instance stop "${SUITECRM_INSTANCE}" 2>/dev/null || true
    apptainer instance stop "${MARIADB_INSTANCE}" 2>/dev/null || true
    sleep 3  # give apptainer time to clean up instance files before re-creating

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
    echo "SuiteCRM starting on $(hostname):${SUITECRM_HTTP_PORT} — will save URL once HTTP is up."
}

wait_for_http() {
    local url="http://localhost:${SUITECRM_HTTP_PORT}"
    local log="${_APTY_HOME}/.apptainer/instances/logs/$(hostname)/${USER}/suitecrm.err"
    local max_wait=900
    local waited=0
    local apache_fixed=0
    echo "Waiting for SuiteCRM at ${url} (first boot ~10-15 min, subsequent ~1 min)..."
    echo "  Log: ${log}"
    until curl -sf "${url}" > /dev/null 2>&1; do
        sleep 15
        waited=$((waited + 15))
        echo "  $(date +%H:%M:%S) waiting... (${waited}s / ${max_wait}s)"

        # Detect Bitnami's hardcoded port 8080 conflict and fix it from inside.
        # appinit always writes 8080 regardless of env vars or pre-patched files;
        # we exec in after appinit to patch the live conf/ and restart Apache.
        if [ "${apache_fixed}" -eq 0 ] && [ -f "${log}" ] && \
           grep -qE "Address already in use|AH00072" "${log}" 2>/dev/null; then
            echo "  Bitnami wrote port 8080 — fixing Apache port via exec..."
            if _exec_fix_apache_port; then
                apache_fixed=1
                # Clear stale port-conflict errors so the loop doesn't re-trigger
                > "${log}" 2>/dev/null || true
                echo "  Port fix applied. Continuing to wait for HTTP..."
            else
                echo "ERROR: exec port fix failed." >&2
                echo "  Is the instance still running? Check: apptainer instance list" >&2
                exit 1
            fi
        fi

        # Print recent log lines every 60s
        if [ $((waited % 60)) -eq 0 ] && [ -f "${log}" ]; then
            echo "  -- last log lines --"
            tail -5 "${log}" | sed 's/^/  /'
            echo "  --------------------"
        fi

        if [ "${waited}" -ge "${max_wait}" ]; then
            echo "ERROR: SuiteCRM did not respond after ${max_wait}s." >&2
            [ -f "${log}" ] && { echo "Last 20 log lines:" >&2; tail -20 "${log}" >&2; }
            exit 1
        fi
    done
    local final_url="http://$(hostname):${SUITECRM_HTTP_PORT}/public"
    printf 'WA_SUITECRM=%s\n' "${final_url}" > "${_SCRATCH}/icrl_wa_env"
    echo "SuiteCRM is up at ${final_url}"
    echo "  WA_SUITECRM=${final_url}"
    echo "  → saved to ${_SCRATCH}/icrl_wa_env"
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
        # Move instead of rm -rf: if a previous container failed mid-cleanup,
        # bind-mounts under the sandbox make rm fail with "Device or resource busy".
        # Rename sidesteps that; the .old dir can be deleted later (or next boot).
        if [ -e "${SUITECRM_SANDBOX}" ]; then
            _OLD="${SUITECRM_SANDBOX}.old"
            rm -rf "${_OLD}" 2>/dev/null || true
            mv "${SUITECRM_SANDBOX}" "${_OLD}" || {
                echo "ERROR: could not move ${SUITECRM_SANDBOX} — kill lingering apptainer processes:" >&2
                echo "  fuser -k ${SUITECRM_SANDBOX}" >&2
                exit 1
            }
        fi
        build_sandbox
        echo "Done. Run 'bash scripts/start_suitecrm_apptainer.sh' to start."
        ;;
    --fresh-install)
        echo "Wiping SuiteCRM data (app + mariadb) for clean reinstall..."
        stop_instances 2>/dev/null || true
        rm -rf "${SUITECRM_DATA}"
        mkdir -p "${SUITECRM_DATA}/mariadb" "${SUITECRM_DATA}/app"
        echo "Data wiped. Starting fresh install (first boot takes ~15 min)..."
        start_instances
        wait_for_http
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
