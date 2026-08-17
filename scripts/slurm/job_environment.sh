# ABOUTME: Shared SLURM job environment: repo/scratch paths, module loading with fallbacks, venv activation
# ABOUTME: Not run directly — sourced by every scripts/slurm/*_job.sh via source scripts/slurm/job_environment.sh
# Shared SLURM environment — source from every scripts/slurm/*.sh script.
# Usage: source "$(dirname "$0")/job_environment.sh"

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
    echo "       Create it with: bash scripts/setup_cluster.sh" >&2
    echo "       Or point ICRL_VENV at an existing one." >&2
    return 1 2>/dev/null || exit 1
fi
# shellcheck disable=SC1091
source "${ICRL_VENV}/bin/activate"
# shellcheck disable=SC1091
[ -f "${ICRL_VENV}/bin/activate_icrl.sh" ] && source "${ICRL_VENV}/bin/activate_icrl.sh"

export STWEBAGENT_ROOT

# The repo ROOT must be on PYTHONPATH: the pipeline imports `src.icrl_dual_training...`,
# `src.lagrangian_finetuning...` etc. as a package. Adding only `src/` puts the *contents* of
# the package on the path and every `from src.x import y` fails.
# STWEBAGENT_ROOT holds the `stwebagentbench` package (task configs + evaluators),
# which is separate from the pip-installed `browsergym.stwebagentbench` shim.
export PYTHONPATH="${ICRL_REPO}:${ICRL_REPO}/src/gridworld:${STWEBAGENT_ROOT}:${PYTHONPATH:-}"

# Export .env so every stage sees WA_SUITECRM / GITLAB / SHOPPING_ADMIN. Only the
# collection scripts call python-dotenv; fine-tuning and CuP evaluation roll out
# real browser episodes and need the same URLs.
if [ -f "${ICRL_REPO}/.env" ]; then
    set -a
    # shellcheck disable=SC1091
    . "${ICRL_REPO}/.env"
    set +a
fi

# ── SuiteCRM shard selection ──────────────────────────────────────────────────
# Throughput against ONE SuiteCRM is capped by the app, not by GPUs: every pass
# that proves completion from the database needs the database to itself. Running
# K independent SuiteCRM+MariaDB instances (scripts/start_suitecrm_shards.sh)
# lifts that cap, and a job joins one by number.
#
# This MUST come after .env is sourced: .env carries the single-instance
# WA_SUITECRM and the shard file has to win. It is also why sharding is a
# per-PROCESS setting — the benchmark copies WA_SUITECRM into SUITECRM and its
# env_config module caches that value at first import, so one process can only
# ever talk to one instance.
ICRL_SUITECRM_SHARD="${ICRL_SUITECRM_SHARD:-}"
if [ -n "${ICRL_SUITECRM_SHARD}" ]; then
    ICRL_SHARD_ENV="${SCRATCH}/suitecrm_shards/shard_${ICRL_SUITECRM_SHARD}.env"
    if [ ! -f "${ICRL_SHARD_ENV}" ]; then
        echo "ERROR: no shard file at ${ICRL_SHARD_ENV}" >&2
        echo "       Start the shards first, on the login node:" >&2
        echo "       bash scripts/start_suitecrm_shards.sh --shards N" >&2
        return 1 2>/dev/null || exit 1
    fi
    set -a
    # shellcheck disable=SC1090
    . "${ICRL_SHARD_ENV}"
    set +a
    echo "suitecrm: shard ${ICRL_SUITECRM_SHARD} → ${WA_SUITECRM}"
fi
export ICRL_SUITECRM_SHARD

# ── Mutual exclusion against the shared web app ───────────────────────────────
# A pass that reseeds SuiteCRM and then attributes database state to its own
# episodes cannot share that instance with anything else. Two passes at once
# truncate each other's records mid-episode and produce traces that look fine
# and prove nothing. The lock is per SHARD, so K shards give K concurrent
# passes while still serialising within each.
#
# mkdir is atomic over NFS and never blocks. `flock -w` was tried and does NOT
# fail fast on this /scratch (observed blocking >10 min), which idles a GPU
# allocation and stalls every afterany-dependent pass behind it. Fast-fail also
# means the same chain can be submitted to several partitions and whichever
# starts first does the work.
ICRL_SUITECRM_LOCK="${ICRL_SUITECRM_LOCK:-${SCRATCH}/locks/suitecrm${ICRL_SUITECRM_SHARD:+_shard_${ICRL_SUITECRM_SHARD}}.lockdir}"

# Bash keeps ONE EXIT trap, so everything that must be cleaned up goes through
# this single handler. (A second `trap ... EXIT` silently replaces the first —
# that is how the lock directory used to survive a job and wedge every later
# pass into exiting immediately.)
ICRL_CLEANUP_PIDS=()
icrl_cleanup() {
    for _pid in "${ICRL_CLEANUP_PIDS[@]:-}"; do kill "${_pid}" 2>/dev/null || true; done
    [ -n "${ICRL_LOCK_HELD:-}" ] && rm -rf "${ICRL_SUITECRM_LOCK}"
    return 0
}

icrl_take_suitecrm_lock() {
    mkdir -p "$(dirname "${ICRL_SUITECRM_LOCK}")"
    if mkdir "${ICRL_SUITECRM_LOCK}" 2>/dev/null; then
        echo "${SLURM_JOB_ID:-local}" > "${ICRL_SUITECRM_LOCK}/owner"
        ICRL_LOCK_HELD=1
        trap icrl_cleanup EXIT
        return 0
    fi
    # Held — but by a job that still exists? A pass killed before its trap ran
    # would otherwise wedge every later pass into exiting immediately.
    local owner
    owner="$(cat "${ICRL_SUITECRM_LOCK}/owner" 2>/dev/null || true)"
    if [ -n "${owner}" ] && ! squeue -h -j "${owner}" -o %i 2>/dev/null | grep -q .; then
        echo "[$(date)] lock owned by job ${owner}, which is gone — reclaiming"
        rm -rf "${ICRL_SUITECRM_LOCK}"
        if mkdir "${ICRL_SUITECRM_LOCK}" 2>/dev/null; then
            echo "${SLURM_JOB_ID:-local}" > "${ICRL_SUITECRM_LOCK}/owner"
            ICRL_LOCK_HELD=1
            trap icrl_cleanup EXIT
            return 0
        fi
    fi
    return 1
}

# Model weights on fast scratch, not NFS home. Compute nodes are offline, so the
# cache must already be populated — see scripts/prefetch_models.py.
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
