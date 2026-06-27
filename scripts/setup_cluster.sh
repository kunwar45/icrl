#!/usr/bin/env bash
# One-time cluster setup for icrl + BrowserGym + ST-WebAgentBench.
#
# Tested on Compute Canada / Alliance clusters (module python/3.12).
#
# Usage (login node):
#   export GITHUB_USER=kunwar45
#   export REPOS_ROOT=$HOME                  # or /scratch/$USER
#   export ICRL_ROOT=$HOME/icrl
#   bash $ICRL_ROOT/scripts/setup_cluster.sh
#
# Optional overrides:
#   VENV_PATH=/scratch/$USER/venvs/icrl_v4
#   SKIP_PLAYWRIGHT=1                          # skip browser download on login node
#   SKIP_MODULES=1                             # skip module load (already in env)
set -euo pipefail

GITHUB_USER="${GITHUB_USER:-}"
REPOS_ROOT="${REPOS_ROOT:-$HOME}"
ICRL_ROOT="${ICRL_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
VENV_PATH="${VENV_PATH:-/scratch/${USER}/venvs/icrl_v4}"
BROWSERGYM_ROOT="${REPOS_ROOT}/BrowserGym"
STWEB_ROOT="${REPOS_ROOT}/ST-WebAgentBench"
SKIP_PLAYWRIGHT="${SKIP_PLAYWRIGHT:-0}"
SKIP_MODULES="${SKIP_MODULES:-0}"

log() { echo "[setup] $*"; }
die() { echo "[setup] ERROR: $*" >&2; exit 1; }

[ -n "$GITHUB_USER" ] || die "Set GITHUB_USER (your GitHub username)."

# ── 1. Cluster modules ────────────────────────────────────────────────────────
if [ "$SKIP_MODULES" != "1" ]; then
    log "Loading cluster modules..."
    module load gcc python/3.12 2>/dev/null || module load python/3.12 2>/dev/null || true
fi

# ── 2. Fork remotes + clone ───────────────────────────────────────────────────
log "Configuring fork remotes..."
bash "${ICRL_ROOT}/scripts/setup_fork_remotes.sh"

# ── 3. Python venv ────────────────────────────────────────────────────────────
if [ ! -d "$VENV_PATH" ]; then
    log "Creating venv at $VENV_PATH"
    mkdir -p "$(dirname "$VENV_PATH")"
    python -m venv "$VENV_PATH"
fi
# shellcheck disable=SC1091
source "${VENV_PATH}/bin/activate"
log "Using venv: $VENV_PATH"

pip install --upgrade pip wheel

# ── 4. icrl package + core requirements ───────────────────────────────────────
log "Installing icrl requirements..."
pip install -e "${ICRL_ROOT}"

# Alliance/Killarney: the computecanada vllm wheels are broken (invalid .dist-info).
# Install everything else first, then install vllm from PyPI separately.
# Requires: module load gcc arrow opencv BEFORE activating the venv.
_TMP_REQS=$(mktemp)
grep -v '^vllm$' "${ICRL_ROOT}/requirements.txt" > "${_TMP_REQS}"
pip install -r "${_TMP_REQS}"
rm -f "${_TMP_REQS}"

log "Installing vllm..."
if pip install --no-index vllm 2>/dev/null; then
    log "vllm installed from cluster wheelhouse."
else
    log "Cluster vllm wheel unavailable or broken — falling back to PyPI."
    log "  Alliance prerequisite: module load gcc arrow opencv (before activating venv)"
    pip install vllm --index-url https://pypi.org/simple/ || \
        log "WARN: vllm install failed. Re-run after: deactivate && module load gcc arrow opencv && source ${VENV_PATH}/bin/activate"
fi

# ── 5. BrowserGym (editable, from fork) ───────────────────────────────────────
log "Installing BrowserGym core..."
pip install -e "${BROWSERGYM_ROOT}/browsergym/core" --no-deps

# ── 6. ST-WebAgentBench (editable integration + root package) ─────────────────
log "Installing ST-WebAgentBench..."
pip install -e "${STWEB_ROOT}/browsergym/stwebagentbench" --no-deps

# ST-WebAgentBench/requirements.txt pins browsergym==0.7.0 which requires
# playwright<1.40. Alliance clusters only ship playwright>=1.57, so skip that
# pin — we already have a compatible browsergym from our BrowserGym fork.
_TMP_STWEB=$(mktemp)
grep -v '^browsergym' "${STWEB_ROOT}/requirements.txt" > "${_TMP_STWEB}"
pip install -r "${_TMP_STWEB}"
rm -f "${_TMP_STWEB}"

# ST-WebAgentBench's requirements may pull an older browsergym-core from PyPI.
# Reinstall our version from the local fork to win.
pip install -e "${BROWSERGYM_ROOT}/browsergym/core" --no-deps

# Make stwebagentbench root importable (custom_env, evaluators, etc.)
SITE_PACKAGES="$(python -c 'import site; print(site.getsitepackages()[0])')"
echo "$(realpath "$STWEB_ROOT")" > "${SITE_PACKAGES}/stwebagentbench.pth"
log "Wrote ${SITE_PACKAGES}/stwebagentbench.pth"

# ── 7. NLTK data ──────────────────────────────────────────────────────────────
log "Downloading NLTK tokenizer data..."
python -c "import nltk; nltk.download('punkt', quiet=True); nltk.download('punkt_tab', quiet=True)"

# ── 8. Playwright Chromium ────────────────────────────────────────────────────
if [ "$SKIP_PLAYWRIGHT" != "1" ]; then
    log "Installing Playwright Chromium (needed on compute nodes for collection)..."
    playwright install chromium
else
    log "Skipping Playwright (SKIP_PLAYWRIGHT=1). Run 'playwright install chromium' before browser jobs."
fi

# ── 9. Environment file ───────────────────────────────────────────────────────
if [ ! -f "${ICRL_ROOT}/.env" ]; then
    cp "${ICRL_ROOT}/.env.example" "${ICRL_ROOT}/.env"
    log "Created ${ICRL_ROOT}/.env — fill in API keys and benchmark URLs."
else
    log ".env already exists — not overwriting."
fi

# Link ST-WebAgentBench .env if missing (benchmark reads WA_SUITECRM etc.)
if [ ! -f "${STWEB_ROOT}/.env" ] && [ -f "${STWEB_ROOT}/.env.example" ]; then
    cp "${STWEB_ROOT}/.env.example" "${STWEB_ROOT}/.env"
    log "Created ${STWEB_ROOT}/.env — set WA_SUITECRM and other web-app URLs."
fi

# ── 10. Shell exports (append to activation helper) ───────────────────────────
ACTIVATE_SNIPPET="${VENV_PATH}/bin/activate_icrl.sh"
cat > "$ACTIVATE_SNIPPET" <<EOF
# Source after: source ${VENV_PATH}/bin/activate
export ICRL_ROOT="${ICRL_ROOT}"
export REPOS_ROOT="${REPOS_ROOT}"
export STWEBAGENT_ROOT="${STWEB_ROOT}"
export BROWSERGYM_ROOT="${BROWSERGYM_ROOT}"
export PYTHONPATH="\${ICRL_ROOT}/gridworld:\${ICRL_ROOT}/src:\${PYTHONPATH:-}"
EOF
log "Wrote ${ACTIVATE_SNIPPET}"
log "After activating the venv, also run: source ${ACTIVATE_SNIPPET}"

# ── 11. Verify ──────────────────────────────────────────────────────────────────
log "Verifying imports..."
export PYTHONPATH="${ICRL_ROOT}/gridworld:${ICRL_ROOT}/src"
python -c "
import browsergym.stwebagentbench, gymnasium as gym
n = len([e for e in gym.envs.registry if 'STWebAgent' in e])
print(f'  ST-WebAgentBench tasks registered: {n}')
assert n > 0, 'No STWebAgent tasks found'
import icrl.envs.stwebagent
print('  icrl.envs.stwebagent: OK')
"

log ""
log "=== Setup complete ==="
log "Next steps:"
log "  1. Edit ${ICRL_ROOT}/.env (API keys)"
log "  2. Edit ${STWEB_ROOT}/.env (WA_SUITECRM=http://...)"
log "  3. source ${VENV_PATH}/bin/activate && source ${ACTIVATE_SNIPPET}"
log "  4. python ${ICRL_ROOT}/scripts/smoke_collection.py"
log "  5. For live browser on cluster: bash scripts/start_suitecrm_apptainer.sh --wait, set WA_SUITECRM in .env"
