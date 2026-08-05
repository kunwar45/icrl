# Shared SLURM environment — source from every slurm/*.sh script.
# Usage: source "$(dirname "$0")/env.sh"

ICRL_REPO="${SLURM_SUBMIT_DIR:-$(pwd)}"
SCRATCH="${SCRATCH:-/scratch/${USER}}"
ICRL_VENV="${ICRL_VENV:-${SCRATCH}/venvs/icrl_v4}"
REPOS_ROOT="${REPOS_ROOT:-${SCRATCH}}"
STWEBAGENT_ROOT="${STWEBAGENT_ROOT:-${REPOS_ROOT}/ST-WebAgentBench}"

# ── Modules BEFORE the venv ───────────────────────────────────────────────────
# Loading a module rewrites PATH, so doing it after activation can shadow the
# venv's python. Each module is loaded separately with a fallback: as one
# `module load a b c` line, a single version that does not exist on this cluster
# makes the whole command fail and NOTHING gets loaded — including python.
icrl_load_module() {
    for candidate in "$@"; do
        if module load "${candidate}" >/dev/null 2>&1; then
            echo "  module: ${candidate}"
            return 0
        fi
    done
    echo "  module: none of [$*] available — continuing"
    return 0
}

# A venv's bin/python is a symlink into the interpreter that built it, so
# loading a *different* python module leaves it dangling. pyvenv.cfg records the
# exact version in plain text and stays readable even when that interpreter is
# not on PATH — read it and ask for that module first. Clusters name modules by
# full version (python/3.12.4); Lmod does not prefix-match "python/3.12".
ICRL_VENV_PY=""
if [ -f "${ICRL_VENV}/pyvenv.cfg" ]; then
    ICRL_VENV_PY=$(awk -F'=' '/^[[:space:]]*version[[:space:]]*=/ {gsub(/[[:space:]]/,"",$2); print $2; exit}' \
                   "${ICRL_VENV}/pyvenv.cfg")
fi

if command -v module >/dev/null 2>&1; then
    echo "Loading modules:"
    icrl_load_module StdEnv/2023 StdEnv/2020
    icrl_load_module gcc
    icrl_load_module ${ICRL_VENV_PY:+"python/${ICRL_VENV_PY}" "python/${ICRL_VENV_PY%.*}"} \
                     python/3.12.4 python/3.12 python/3.11.5 python/3.11 python
    icrl_load_module cuda/12.6 cuda/12.9 cuda/12.2 cuda
    icrl_load_module arrow
    icrl_load_module apptainer/1.4.5 apptainer/1.3.5 apptainer
fi

# ── Virtualenv ────────────────────────────────────────────────────────────────
if [ ! -f "${ICRL_VENV}/bin/activate" ]; then
    echo "ERROR: no virtualenv at ${ICRL_VENV}" >&2
    echo "       Create it with: bash scripts/infra/setup_cluster.sh" >&2
    echo "       Or point ICRL_VENV at an existing one." >&2
    return 1 2>/dev/null || exit 1
fi
# shellcheck disable=SC1091
source "${ICRL_VENV}/bin/activate"
# shellcheck disable=SC1091
[ -f "${ICRL_VENV}/bin/activate_icrl.sh" ] && source "${ICRL_VENV}/bin/activate_icrl.sh"

export STWEBAGENT_ROOT

# The repo ROOT must be on PYTHONPATH: the pipeline imports `src.constraint...`,
# `src.finetune...` etc. as a package. Adding only `src/` puts the *contents* of
# the package on the path and every `from src.x import y` fails.
# STWEBAGENT_ROOT holds the `stwebagentbench` package (task configs + evaluators),
# which is separate from the pip-installed `browsergym.stwebagentbench` shim.
export PYTHONPATH="${ICRL_REPO}:${ICRL_REPO}/src:${STWEBAGENT_ROOT}:${PYTHONPATH:-}"

# Export .env so every stage sees WA_SUITECRM / GITLAB / SHOPPING_ADMIN. Only the
# collection scripts call python-dotenv; fine-tuning and CuP evaluation roll out
# real browser episodes and need the same URLs.
if [ -f "${ICRL_REPO}/.env" ]; then
    set -a
    # shellcheck disable=SC1091
    . "${ICRL_REPO}/.env"
    set +a
fi

# Model weights on fast scratch, not NFS home. Compute nodes are offline, so the
# cache must already be populated — see scripts/infra/prefetch_models.py.
export HF_HOME="${HF_HOME:-${SCRATCH}/hf_cache}"

echo "venv    : ${ICRL_VENV}"
echo "python  : $(command -v python) ($(python -V 2>&1))"

# A venv whose interpreter vanished reports nothing useful later on; catch it here.
if ! python -c "import sys" >/dev/null 2>&1; then
    echo "ERROR: the venv's python does not run — the module that built it" >&2
    echo "       (python/${ICRL_VENV_PY:-unknown}) is probably not loaded." >&2
    echo "       Check: module avail python" >&2
    return 1 2>/dev/null || exit 1
fi
