#!/usr/bin/env bash
# Installs SuiteCRM 8 directly via the Symfony console, bypassing Bitnami's
# broken wizard. Run this ONCE after pre-seeding the app dir with the
# SuiteCRM 8 release zip (SuiteCRM-8.10.1.zip extracted to $SUITECRM_DATA/app/).
#
# Usage:
#   bash scripts/install_suitecrm_direct.sh
#
# Prerequisites:
#   1. MariaDB SIF: $SCRATCH/apptainer/mariadb.sif
#   2. SuiteCRM sandbox: $SCRATCH/apptainer/suitecrm_sandbox
#   3. SuiteCRM 8 files extracted to: $SCRATCH/suitecrm/app/
#      (should contain bin/console, public/, etc.)
set -euo pipefail

_SCRATCH="${SCRATCH:-/scratch/${USER}}"
_APTY_HOME="${_SCRATCH}"
mkdir -p "${_APTY_HOME}/.apptainer"
apptainer() { HOME="${_APTY_HOME}" command apptainer "$@"; }

MARIADB_SIF="${MARIADB_SIF:-${_SCRATCH}/apptainer/mariadb.sif}"
SUITECRM_SANDBOX="${SUITECRM_SANDBOX:-${_SCRATCH}/apptainer/suitecrm_sandbox}"
SUITECRM_DATA="${SUITECRM_DATA:-${_SCRATCH}/suitecrm}"
MARIADB_INSTANCE="${MARIADB_INSTANCE:-mariadb}"
PORT="${SUITECRM_HTTP_PORT:-8080}"

_php="${SUITECRM_SANDBOX}/opt/bitnami/php/bin/php"
_console="/bitnami/suitecrm/bin/console"

# ── Sanity checks ─────────────────────────────────────────────────────────────
if [ ! -f "${SUITECRM_DATA}/app/bin/console" ]; then
    echo "ERROR: ${SUITECRM_DATA}/app/bin/console not found." >&2
    echo "  Unzip SuiteCRM-8.10.1.zip to ${SUITECRM_DATA}/app/ first." >&2
    exit 1
fi

# ── Load apptainer module if needed ───────────────────────────────────────────
if ! command -v apptainer &>/dev/null; then
    for _f in /cvmfs/soft.computecanada.ca/nix/var/nix/profiles/16.09/lmod/lmod/init/bash \
               /etc/profile.d/lmod.sh; do
        [ -f "$_f" ] && source "$_f" 2>/dev/null && break
    done
    module load apptainer/1.4.5 2>/dev/null || true
fi

# ── Start MariaDB if not running ──────────────────────────────────────────────
if ! apptainer instance list 2>/dev/null | grep -q "^${MARIADB_INSTANCE}"; then
    echo "Starting MariaDB..."
    mkdir -p "${SUITECRM_DATA}/mariadb"
    apptainer instance stop "${MARIADB_INSTANCE}" 2>/dev/null || true
    sleep 2
    apptainer instance run \
        --writable-tmpfs \
        --bind "${SUITECRM_DATA}/mariadb:/bitnami/mariadb" \
        --env ALLOW_EMPTY_PASSWORD=yes \
        --env MARIADB_USER=bn_suitecrm \
        --env MARIADB_DATABASE=bitnami_suitecrm \
        --env MARIADB_PASSWORD=bitnami123 \
        "${MARIADB_SIF}" "${MARIADB_INSTANCE}"
fi

echo "Waiting for MariaDB (up to 120s)..."
_waited=0
until apptainer exec instance://"${MARIADB_INSTANCE}" \
    mysql -ubn_suitecrm -pbitnami123 bitnami_suitecrm -e "SELECT 1" >/dev/null 2>&1; do
    sleep 5; _waited=$((_waited+5))
    echo "  ${_waited}s..."
    [ "${_waited}" -ge 120 ] && { echo "ERROR: MariaDB timeout" >&2; exit 1; }
done
echo "  MariaDB ready."

# ── Show installer help (uncomment to inspect args) ───────────────────────────
# apptainer exec -w -B "${SUITECRM_DATA}/app:/bitnami/suitecrm" \
#     "${SUITECRM_SANDBOX}" "${_php}" "${_console}" suitecrm:app:install --help

# ── Run suitecrm:app:install ──────────────────────────────────────────────────
echo "Running suitecrm:app:install (this takes ~5 min)..."
apptainer exec -w \
    -B "${SUITECRM_DATA}/app:/bitnami/suitecrm" \
    "${SUITECRM_SANDBOX}" \
    "${_php}" "${_console}" suitecrm:app:install \
        --site_host="http://$(hostname):${PORT}" \
        --db_host=127.0.0.1 \
        --db_port=3306 \
        --db_name=bitnami_suitecrm \
        --db_username=bn_suitecrm \
        --db_password=bitnami123 \
        --site_username=admin \
        --site_password=Admin1234! \
        --site_email=admin@example.com \
        --no-interaction 2>&1

echo "Install command exited with status $?"

# ── Check if install succeeded ────────────────────────────────────────────────
if [ ! -f "${SUITECRM_DATA}/app/config/services/local.api.params.php" ] && \
   [ ! -f "${SUITECRM_DATA}/app/config_si.php" ] && \
   [ ! -f "${SUITECRM_DATA}/app/config.php" ]; then
    echo "WARNING: No config file found — install may have failed. Check output above." >&2
else
    echo "Config file found — install appears successful."
fi

# ── Create Bitnami initialized marker ─────────────────────────────────────────
touch "${SUITECRM_DATA}/app/.initialized" 2>/dev/null || true
echo "Marked as initialized."

# ── Start the SuiteCRM instance (Bitnami detects .initialized → restore path) ─
echo ""
echo "Starting SuiteCRM instance..."
apptainer instance stop suitecrm 2>/dev/null || true
sleep 2
apptainer instance run \
    --writable \
    -B "${SUITECRM_DATA}/app:/bitnami/suitecrm" \
    --env APACHE_HTTP_PORT_NUMBER="${PORT}" \
    --env APACHE_HTTPS_PORT_NUMBER="$((PORT+1))" \
    --env SUITECRM_DATABASE_HOST=127.0.0.1 \
    --env SUITECRM_DATABASE_PORT_NUMBER=3306 \
    --env SUITECRM_DATABASE_USER=bn_suitecrm \
    --env SUITECRM_DATABASE_NAME=bitnami_suitecrm \
    --env SUITECRM_DATABASE_PASSWORD=bitnami123 \
    --env SUITECRM_USERNAME=admin \
    --env SUITECRM_PASSWORD=Admin1234! \
    --env SUITECRM_HOST="$(hostname)" \
    --env ALLOW_EMPTY_PASSWORD=yes \
    "${SUITECRM_SANDBOX}" suitecrm

echo "Waiting for HTTP on port ${PORT}..."
_waited=0
until curl -sf "http://localhost:${PORT}" >/dev/null 2>&1; do
    sleep 10; _waited=$((_waited+10))
    echo "  ${_waited}s..."
    [ "${_waited}" -ge 300 ] && { echo "Timeout waiting for HTTP" >&2; exit 1; }
done

_url="http://$(hostname):${PORT}/index.php"
printf 'WA_SUITECRM=%s\n' "${_url}" > "${_SCRATCH}/icrl_wa_env"
echo "SuiteCRM is up at ${_url}"
echo "  Login: admin / Admin1234!"
echo "  WA_SUITECRM saved to ${_SCRATCH}/icrl_wa_env"
