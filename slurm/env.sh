# Shared SLURM environment — source from every slurm/*.sh script.
# Usage: source "$(dirname "$0")/env.sh"

ICRL_VENV="${ICRL_VENV:-/scratch/${USER}/venvs/icrl_v4}"
REPOS_ROOT="${REPOS_ROOT:-/scratch/${USER}}"
STWEBAGENT_ROOT="${STWEBAGENT_ROOT:-${REPOS_ROOT}/ST-WebAgentBench}"

# shellcheck disable=SC1091
source "${ICRL_VENV}/bin/activate"
# shellcheck disable=SC1091
[ -f "${ICRL_VENV}/bin/activate_icrl.sh" ] && source "${ICRL_VENV}/bin/activate_icrl.sh"

export STWEBAGENT_ROOT
export PYTHONPATH="${SLURM_SUBMIT_DIR:-$(pwd)}/gridworld:${SLURM_SUBMIT_DIR:-$(pwd)}/src:${PYTHONPATH:-}"

module load gcc python/3.12 arrow/23.0.1 cuda/12.1 cudnn/8.9 2>/dev/null || true

# Apptainer is optional — only needed when starting SuiteCRM inline in gen_safe_demos.sh.
