#!/usr/bin/env bash
# Source this at the start of every SSH session on the login node.
#
# Usage:
#   source ~/projects/icrl/scripts/session_start.sh
#
# What it does:
#   1. Loads cluster modules (apptainer, cuda, etc.)
#   2. Activates the Python venv
#   3. Sets PYTHONPATH / STWEBAGENT_ROOT
#   4. Checks if SuiteCRM Apptainer instances are running; starts them if not
#   5. Loads WA_SUITECRM into your shell so jobs can pick it up
#   6. Prints a status summary

# ── Guard: must be sourced, not executed ─────────────────────────────────────
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    echo "ERROR: source this script, don't execute it:"
    echo "  source ${BASH_SOURCE[0]}"
    exit 1
fi

_ICRL_ROOT="${ICRL_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
_VENV="${ICRL_VENV:-/scratch/${USER}/venvs/icrl_v4}"
_STWEB_ROOT="${STWEBAGENT_ROOT:-/scratch/${USER}/ST-WebAgentBench}"
_WA_ENV_FILE="/scratch/${USER}/icrl_wa_env"

echo "=== ICRL session startup ==="

# /home Lustre is degraded — redirect apptainer instance state to /scratch.
# Must be set before any apptainer call (including instance list).
_APTY_HOME="/scratch/${USER}"
mkdir -p "${_APTY_HOME}/.apptainer"
apptainer() { HOME="${_APTY_HOME}" command apptainer "$@"; }

# ── 1. Modules ────────────────────────────────────────────────────────────────
echo "[1/5] Loading modules..."
module load gcc python/3.12 arrow/23.0.1 cuda/12.9 cudnn apptainer/1.4.5 2>/dev/null || true
echo "      OK"

# ── 2. Venv ───────────────────────────────────────────────────────────────────
echo "[2/5] Activating venv: ${_VENV}"
if [ ! -f "${_VENV}/bin/activate" ]; then
    echo "      ERROR: venv not found at ${_VENV}"
    echo "      Run setup first: bash ${_ICRL_ROOT}/scripts/setup_cluster.sh"
    return 1
fi
# shellcheck disable=SC1090
source "${_VENV}/bin/activate"
[ -f "${_VENV}/bin/activate_icrl.sh" ] && source "${_VENV}/bin/activate_icrl.sh"
echo "      OK (python: $(python --version 2>&1))"

# ── 3. PYTHONPATH ─────────────────────────────────────────────────────────────
export ICRL_ROOT="${_ICRL_ROOT}"
export STWEBAGENT_ROOT="${_STWEB_ROOT}"
export PYTHONPATH="${_ICRL_ROOT}/gridworld:${_ICRL_ROOT}/src:${PYTHONPATH:-}"
echo "[3/5] PYTHONPATH set"

# ── 4. SuiteCRM / Apptainer ───────────────────────────────────────────────────
echo "[4/5] Checking SuiteCRM Apptainer instances..."
_RUNNING=$(apptainer instance list 2>/dev/null | awk 'NR>1 {print $1}')
_HAS_MARIADB=$(echo "$_RUNNING" | grep -c '^mariadb$' || true)
_HAS_SUITECRM=$(echo "$_RUNNING" | grep -c '^suitecrm$' || true)

if [ "$_HAS_MARIADB" -gt 0 ] && [ "$_HAS_SUITECRM" -gt 0 ]; then
    echo "      Both instances already running."
else
    echo "      Instances not running — starting SuiteCRM (this may take ~1 min)..."
    bash "${_ICRL_ROOT}/scripts/start_suitecrm_apptainer.sh"
fi

# ── 5. WA_SUITECRM env var ────────────────────────────────────────────────────
echo "[5/5] Loading WA_SUITECRM..."
if [ -f "${_WA_ENV_FILE}" ]; then
    # shellcheck disable=SC1090
    export WA_SUITECRM
    WA_SUITECRM="$(grep '^WA_SUITECRM=' "${_WA_ENV_FILE}" | cut -d= -f2-)"
    export SUITECRM="${WA_SUITECRM}"
    echo "      WA_SUITECRM=${WA_SUITECRM}"
else
    echo "      WARNING: ${_WA_ENV_FILE} not found."
    echo "      Run: bash ${_ICRL_ROOT}/scripts/start_suitecrm_apptainer.sh"
fi

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
echo "=== Ready ==="
echo "  venv     : ${_VENV}"
echo "  icrl     : ${_ICRL_ROOT}"
echo "  ST-Web   : ${_STWEB_ROOT}"
echo "  SuiteCRM : ${WA_SUITECRM:-NOT SET}"
echo ""
echo "To submit a job:  sbatch ${_ICRL_ROOT}/slurm/gen_safe_demos.sh"
echo "To check status:  squeue -u \$USER"
