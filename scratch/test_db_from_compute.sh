#!/bin/bash
# ABOUTME: Confirms a SLURM compute node can reach the SuiteCRM database for the state-verification gate.
# ABOUTME: Run: sbatch --account=... scripts/slurm/... (submitted ad hoc; 1 CPU, no GPU, ~1 min)
#SBATCH --job-name=icrl-db-reachability
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:05:00
#SBATCH --output=logs/slurm/%x_%j.out
#SBATCH --error=logs/slurm/%x_%j.err
set -uo pipefail
cd "${SLURM_SUBMIT_DIR}"
source scripts/slurm/job_environment.sh
set -a; source .env; set +a

echo "compute node: $(hostname)"
python - <<'PY'
import os, socket, sys
host = os.environ.get("ICRL_SUITECRM_DB_HOST", "klogin01")
port = int(os.environ.get("ICRL_SUITECRM_DB_PORT", "3306"))
try:
    with socket.create_connection((host, port), timeout=10):
        print(f"TCP {host}:{port} REACHABLE")
except Exception as e:
    print(f"TCP {host}:{port} UNREACHABLE: {e}")
    sys.exit(1)

sys.path.insert(0, os.getcwd())
from src.trajectory_collection.stwebagentbench_state_verifier import verify_task_state
ok, detail = verify_task_state(235)
print(f"state check ran from compute node: persisted={ok}")
print(detail)
print("GATE USABLE FROM COMPUTE NODE")
PY
