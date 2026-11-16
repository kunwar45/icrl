#!/bin/bash
# ABOUTME: Submits a secret-keeping pipeline job and, by default, publishes the run to private Hugging Face repos from the login node once the job completes (compute nodes have no internet)
# ABOUTME: Run on a fir or killarney login node from the repo root: CONFIG=configs/lagrangian_finetuning/<config>.yaml scripts/submit_secret_keeping_run.sh [extra sbatch args]   (PUBLISH=0 to skip publishing)
set -euo pipefail
cd "$(dirname "$0")/.."
: "${CONFIG:?set CONFIG to a configs/lagrangian_finetuning/secret_keeping_*.yaml file}"
# Per-cluster defaults; any of these can be overridden from the environment.
case "$(hostname -f 2>/dev/null || hostname)" in
  *killarney*)
    : "${ICRL_ACCOUNT:=aip-s2ganapa}"
    : "${ICRL_VENV:=/project/aip-s2ganapa/kunwar/venvs/icrl_v4}"
    : "${HF_HOME:=/project/aip-s2ganapa/kunwar/hf_cache}" ;;
  *)  # fir
    : "${ICRL_ACCOUNT:=def-s2ganapa}"
    : "${ICRL_VENV:=$SCRATCH/venvs/icrl_v4}"
    : "${HF_HOME:=$SCRATCH/hf_cache}" ;;
esac
: "${ICRL_GPU:=h100:4}"   # killarney: l40s:4 usually schedules sooner
: "${ICRL_TIME:=06:00:00}"
export ICRL_VENV HF_HOME CONFIG
PY="${ICRL_VENV}/bin/python"
mkdir -p logs/slurm logs/publish

RUN_DIR=$("$PY" -c "
import sys; sys.path.insert(0, '.')
from pathlib import Path
import scripts.run_secret_keeping_icrl_pipeline as P
print(P.load_cfg(Path('${CONFIG}').resolve(), [], smoke=False).output.dir)
")
JOB=$(sbatch --parsable --account="$ICRL_ACCOUNT" --gres="gpu:${ICRL_GPU}" --cpus-per-task=16 --mem=160G \
  --time="$ICRL_TIME" --job-name="$(basename "$CONFIG" .yaml)" --export=ALL "$@" \
  scripts/slurm/run_secret_keeping_icrl_pipeline_job.sh)
echo "submitted job ${JOB} -> ${RUN_DIR}"

if [ "${PUBLISH:-1}" = "0" ]; then
  echo "PUBLISH=0: not publishing; later run: $PY scripts/push_secret_keeping_run_to_huggingface.py --run-dir ${RUN_DIR}"
  exit 0
fi
if ! "$PY" -c "from huggingface_hub import HfApi; HfApi().whoami()" >/dev/null 2>&1; then
  echo "WARNING: no Hugging Face login on this node; run 'hf auth login' once, then publish with:"
  echo "  $PY scripts/push_secret_keeping_run_to_huggingface.py --run-dir ${RUN_DIR}"
  exit 0
fi
# Login-node watcher: poll the scheduler every 5 min, publish if the job completed.
LOG="logs/publish/${JOB}.log"
nohup bash -c "
  while squeue -h -j ${JOB} 2>/dev/null | grep -q .; do sleep 300; done
  STATE=\$(sacct -j ${JOB} -X -n -o State%20 | tr -d ' ')
  echo \"job ${JOB} ended: \${STATE}\"
  if [ \"\${STATE}\" = COMPLETED ]; then
    '$PY' scripts/push_secret_keeping_run_to_huggingface.py --run-dir '${RUN_DIR}'
  else
    echo 'not publishing a job that did not complete'
  fi
" > "$LOG" 2>&1 &
echo "will publish to Hugging Face when the job completes (watcher on $(hostname), log ${LOG})"
