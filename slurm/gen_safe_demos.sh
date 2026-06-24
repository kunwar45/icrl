#!/bin/bash
# ============================================================================
# gen_safe_demos.sh — Collect CuP=1 safe trajectories via vLLM + Qwen-72B
# ============================================================================
#
# Starts a vLLM server on the allocated GPUs, then runs the collection loop
# for ST-WebAgentBench Easy-tier SuiteCRM tasks (IDs 235-254 by default).
#
# ── Quick start ──────────────────────────────────────────────────────────────
#
#   # All 20 easy tasks, 5 retries each:
#   sbatch slurm/gen_safe_demos.sh
#
#   # First N tasks only (smoke test):
#   N_TASKS=3 sbatch slurm/gen_safe_demos.sh
#
#   # Explicit task IDs:
#   TASK_IDS="235 236 237 238 239" sbatch slurm/gen_safe_demos.sh
#
#   # More retries / longer episodes:
#   MAX_RETRIES=10 MAX_STEPS=40 sbatch slurm/gen_safe_demos.sh
#
#   # Dry-run (validate imports, env, vLLM health — no browser):
#   DRY_RUN=1 sbatch slurm/gen_safe_demos.sh
#
# ── Environment variables ────────────────────────────────────────────────────
#
#   N_TASKS        How many easy-tier tasks to run (default: all 20).
#                  Tasks are taken in order from EASY_TASK_IDS (235-254).
#                  Ignored if TASK_IDS is set explicitly.
#
#   TASK_IDS       Space-separated list of task IDs to run.
#                  Example: TASK_IDS="235 236 237"
#
#   MAX_RETRIES    Attempts per task before flagging as failed (default: 5).
#
#   MAX_STEPS      Max browser steps per episode (default: 30).
#
#   OUTPUT_DIR     Where to save trajectory JSON files (default: below).
#
#   MODEL          HuggingFace model ID (default: Qwen/Qwen2.5-72B-Instruct).
#
#   TP_SIZE        vLLM tensor-parallel degree, must match --gres=gpu:N
#                  (default: 4).
#
#   VLLM_PORT      Port for the vLLM OpenAI-compat server (default: 8100).
#
#   DRY_RUN        Set to 1 to run --dry-run only (no browser, no episodes).
#
#   WA_SUITECRM    SuiteCRM URL reachable from the compute node.
#                  If unset the job starts SuiteCRM via Apptainer on the compute
#                  node itself.  Recommended on Alliance: start SuiteCRM once
#                  on the login node (scripts/start_suitecrm_apptainer.sh) and set
#                  WA_SUITECRM=http://<login-node>:8080/public in icrl/.env.
#
#   SUITECRM_DATA  Persistent data dir for SuiteCRM + MariaDB on /scratch.
#                  Default: /scratch/$USER/suitecrm
#                  First run initialises the DB (~5-10 min). Subsequent runs
#                  reuse the existing DB and boot in ~30 s.
#
#   MARIADB_SIF    Apptainer image for MariaDB (default: below).
#   SUITECRM_SIF   Apptainer image for SuiteCRM (default: below).
#                  Pull once with apptainer pull from docker://bitnamilegacy/*.
#                  public.ecr.aws/bitnami/* was removed 2026-06-10.
#
# ── Resource sizing ──────────────────────────────────────────────────────────
#
#   Qwen-72B in bfloat16 ≈ 144 GB VRAM.
#   4× A100-40GB  → set TP_SIZE=4, --gres=gpu:h100:4
#   2× A100-80GB  → set TP_SIZE=2, --gres=gpu:a100l:2   (L = 80 GB variant)
#
#SBATCH --job-name=icrl-gen
#SBATCH --account=def-s2ganapa_gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:h100:4
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=logs/slurm/%x_%j.out
#SBATCH --error=logs/slurm/%x_%j.err

set -euo pipefail

# ── Defaults (override via env vars before sbatch) ───────────────────────────

MODEL="${MODEL:-Qwen/Qwen2.5-72B-Instruct}"
TP_SIZE="${TP_SIZE:-4}"
VLLM_PORT="${VLLM_PORT:-8100}"
VLLM_URL="http://localhost:${VLLM_PORT}/v1"

MAX_RETRIES="${MAX_RETRIES:-5}"
MAX_STEPS="${MAX_STEPS:-30}"

# All 20 easy-tier task IDs in order
ALL_EASY_TASKS="235 236 237 238 239 240 241 242 243 244 245 246 247 248 249 250 251 252 253 254"

OUTPUT_DIR="${OUTPUT_DIR:-${SCRATCH:-$SLURM_SUBMIT_DIR}/trajectories/safe}"
HF_CACHE="${SCRATCH:-/tmp}/hf_cache"
LOG_DIR="logs/slurm"
DRY_RUN="${DRY_RUN:-0}"

SUITECRM_DATA="${SUITECRM_DATA:-/scratch/${USER}/suitecrm}"
MARIADB_SIF="${MARIADB_SIF:-/scratch/${USER}/apptainer/mariadb.sif}"
SUITECRM_SIF="${SUITECRM_SIF:-/scratch/${USER}/apptainer/suitecrm.sif}"
SUITECRM_SANDBOX="${SUITECRM_SANDBOX:-/scratch/${USER}/apptainer/suitecrm_sandbox}"
APPTAINER_TMPDIR="${APPTAINER_TMPDIR:-/scratch/${USER}/apptainer/tmp}"
SUITECRM_OVERLAY="${SUITECRM_OVERLAY:-/scratch/${USER}/apptainer/suitecrm_overlay.img}"

# ── Resolve task list ─────────────────────────────────────────────────────────

if [ -n "${TASK_IDS:-}" ]; then
    # Explicit list takes priority
    RESOLVED_TASKS="${TASK_IDS}"
elif [ -n "${N_TASKS:-}" ]; then
    # Take first N from the easy-tier list
    RESOLVED_TASKS=$(echo "${ALL_EASY_TASKS}" | tr ' ' '\n' | head -n "${N_TASKS}" | tr '\n' ' ')
else
    RESOLVED_TASKS="${ALL_EASY_TASKS}"
fi

N_RESOLVED=$(echo "${RESOLVED_TASKS}" | wc -w | tr -d ' ')

# ── Setup ─────────────────────────────────────────────────────────────────────

# Ensure we run from the project root (same pattern as all other SLURM scripts).
# Relative paths (scripts/, logs/slurm/) depend on this.
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

mkdir -p "${LOG_DIR}" "${OUTPUT_DIR}" "${HF_CACHE}"

export HF_HOME="${HF_CACHE}"
export TRANSFORMERS_CACHE="${HF_CACHE}"

# Load cluster modules if available (no-op outside SLURM)
module load gcc python/3.12 arrow/23.0.1 cuda/12.9 cudnn apptainer/1.4.5 2>/dev/null || true
source /scratch/kunwar/venvs/icrl_v4/bin/activate 2>/dev/null || true

# ── Pre-flight: verify critical imports before wasting GPU time ───────────────
echo "[$(date +%H:%M:%S)] Pre-flight import check..."
python - <<'PREFLIGHT'
import sys
failures = []
checks = [
    # Core stack
    ("vllm",                              "vllm"),
    ("fastapi",                           "fastapi"),
    ("torch",                             "torch"),
    ("playwright",                        "playwright.sync_api"),
    ("dotenv",                            "dotenv"),
    ("browsergym",                        "browsergym"),
    # vLLM API server deps (discovered via past startup crashes)
    ("prometheus_fastapi_instrumentator", "prometheus_fastapi_instrumentator"),
    ("prometheus_client",                 "prometheus_client"),
    ("uvloop",                            "uvloop"),
    ("opentelemetry",                     "opentelemetry.sdk"),
    ("msgspec",                           "msgspec"),
    ("partial_json_parser",               "partial_json_parser"),
    ("xgrammar",                          "xgrammar"),
    ("outlines",                          "outlines"),
    ("interegular",                       "interegular"),
    ("sentencepiece",                     "sentencepiece"),
    ("lark",                              "lark"),
    ("filelock",                          "filelock"),
    ("compressed_tensors",                "compressed_tensors"),
    ("numba",                             "numba"),
]
for label, mod in checks:
    try:
        __import__(mod)
        print(f"  [ok] {label}")
    except ImportError as e:
        print(f"  [MISSING] {label}: {e}", file=sys.stderr)
        failures.append(label)

# Check torch sees CUDA
try:
    import torch
    if not torch.cuda.is_available():
        print("  [WARN] torch.cuda.is_available() = False (expected on login node, fatal on compute)", file=sys.stderr)
    else:
        n = torch.cuda.device_count()
        print(f"  [ok] torch.cuda — {n} GPU(s) visible")
except Exception as e:
    print(f"  [WARN] torch.cuda check failed: {e}", file=sys.stderr)

if failures:
    print(f"\nPre-flight FAILED — missing: {', '.join(failures)}", file=sys.stderr)
    print("Fix: cd icrl && source /scratch/$USER/venvs/icrl_v4/bin/activate && pip install <package>", file=sys.stderr)
    sys.exit(1)
print("Pre-flight passed.")
PREFLIGHT

PREFLIGHT_EXIT=$?
if [ "${PREFLIGHT_EXIT}" -ne 0 ]; then
    echo "ERROR: Pre-flight failed — aborting to avoid wasting GPU time."
    exit "${PREFLIGHT_EXIT}"
fi
echo ""

# WA_SUITECRM fallback: if not already in the environment, try the scratch file
# written by start_suitecrm_apptainer.sh (avoids depending on /home Lustre).
if [ -z "${WA_SUITECRM:-}" ] && [ -f "/scratch/${USER}/icrl_wa_env" ]; then
    WA_SUITECRM="$(grep '^WA_SUITECRM=' "/scratch/${USER}/icrl_wa_env" | cut -d= -f2-)"
    export WA_SUITECRM
    echo "[$(date +%H:%M:%S)] Loaded WA_SUITECRM from /scratch/${USER}/icrl_wa_env: ${WA_SUITECRM}"
fi

# Load .env defaults without overriding vars already set in the environment
# (vars passed via the submitting shell or sbatch --export take precedence)
_DOTENV="${SLURM_SUBMIT_DIR:-.}/.env"
if [ -f "${_DOTENV}" ]; then
    while IFS= read -r _line || [ -n "${_line}" ]; do
        [[ "${_line}" =~ ^[[:space:]]*# ]] && continue
        [[ -z "${_line// }" ]] && continue
        if [[ "${_line}" =~ ^([A-Za-z_][A-Za-z0-9_]*)=(.*) ]]; then
            _var="${BASH_REMATCH[1]}"
            _val="${BASH_REMATCH[2]}"
            # Use a plain variable for indirect expansion — array-subscript
            # indirect expansion (${!BASH_REMATCH[1]+x}) is unreliable across
            # bash versions and silently skips the assignment.
            [[ -z "${!_var:-}" ]] && export "$_var=$_val"
        fi
    done < "${_DOTENV}"
fi
unset _DOTENV _line _var _val

load_apptainer() {
    if command -v apptainer &>/dev/null; then
        echo "[$(date +%H:%M:%S)] apptainer: $(command -v apptainer) ($(apptainer --version 2>/dev/null || echo unknown))"
        return 0
    fi
    for mod in apptainer/1.4.5 apptainer/1.3.5; do
        if module load "$mod" 2>/dev/null && command -v apptainer &>/dev/null; then
            echo "[$(date +%H:%M:%S)] Loaded ${mod} → $(command -v apptainer)"
            return 0
        fi
    done
    echo "ERROR: apptainer not found. Load manually: module load apptainer/1.4.5" >&2
    exit 127
}

echo "========================================================================"
echo " icrl-gen  job ${SLURM_JOB_ID:-local}  node $(hostname)"
echo "========================================================================"
echo "  Model       : ${MODEL}"
echo "  TP size     : ${TP_SIZE}"
echo "  Tasks (${N_RESOLVED}): ${RESOLVED_TASKS}"
echo "  Max retries : ${MAX_RETRIES}"
echo "  Max steps   : ${MAX_STEPS}"
echo "  Output dir  : ${OUTPUT_DIR}"
echo "  HF cache    : ${HF_CACHE}"
echo "========================================================================"
echo ""

# ── SuiteCRM via Apptainer (skipped if WA_SUITECRM already set externally) ────
#
# Recommended: start SuiteCRM once on the login node with start_suitecrm_apptainer.sh
# and set WA_SUITECRM in .env so this block is skipped entirely.
# SuiteCRM uses a persistent ext3 overlay (not --writable-tmpfs) to avoid
# "No space left on device" from the large Angular cache baked into the SIF.

if [ -z "${WA_SUITECRM:-}" ]; then
    if [ "${DRY_RUN}" = "1" ]; then
        export WA_SUITECRM="http://localhost:8080"
        echo "[$(date +%H:%M:%S)] DRY RUN — skipping Apptainer SuiteCRM startup"
    else
    echo "[$(date +%H:%M:%S)] Starting MariaDB + SuiteCRM via Apptainer..."
    load_apptainer
    export APPTAINER_TMPDIR
    mkdir -p "${APPTAINER_TMPDIR}" "${SUITECRM_DATA}/mariadb" "${SUITECRM_DATA}/app"

    for sif in "${MARIADB_SIF}"; do
        if [ ! -f "${sif}" ]; then
            echo "ERROR: missing ${sif}. Pull with:" >&2
            echo "  apptainer pull ${sif} docker://bitnamilegacy/mariadb:11.4" >&2
            exit 1
        fi
    done
    if [ ! -d "${SUITECRM_SANDBOX}" ]; then
        if [ ! -f "${SUITECRM_SIF}" ]; then
            echo "ERROR: missing ${SUITECRM_SIF} and sandbox ${SUITECRM_SANDBOX}." >&2
            exit 1
        fi
        echo "[$(date +%H:%M:%S)] Building SuiteCRM sandbox (one-time, ~2 min)..."
        apptainer build --sandbox "${SUITECRM_SANDBOX}" "${SUITECRM_SIF}"
    fi

    CRM_MARIADB="mariadb_${SLURM_JOB_ID:-local}"
    CRM_APP="suitecrm_${SLURM_JOB_ID:-local}"

    apptainer instance stop "${CRM_APP}" 2>/dev/null || true
    apptainer instance stop "${CRM_MARIADB}" 2>/dev/null || true

    apptainer instance run \
        --writable-tmpfs \
        --bind "${SUITECRM_DATA}/mariadb:/bitnami/mariadb" \
        --env ALLOW_EMPTY_PASSWORD=yes \
        --env MARIADB_USER=bn_suitecrm \
        --env MARIADB_DATABASE=bitnami_suitecrm \
        --env MARIADB_PASSWORD=bitnami123 \
        "${MARIADB_SIF}" "${CRM_MARIADB}"

    echo "[$(date +%H:%M:%S)] Waiting 30 s for MariaDB to initialise..."
    sleep 30

    # Use --writable sandbox: AllowSetuidMountExtfs=false on this cluster means
    # --overlay ext3 fails; --writable-tmpfs fills the 64 MB default tmpfs.
    apptainer instance run \
        --writable \
        --bind "${SUITECRM_DATA}/app:/bitnami/suitecrm" \
        --env SUITECRM_DATABASE_HOST=127.0.0.1 \
        --env SUITECRM_DATABASE_PORT_NUMBER=3306 \
        --env SUITECRM_DATABASE_USER=bn_suitecrm \
        --env SUITECRM_DATABASE_NAME=bitnami_suitecrm \
        --env SUITECRM_DATABASE_PASSWORD=bitnami123 \
        --env ALLOW_EMPTY_PASSWORD=yes \
        "${SUITECRM_SANDBOX}" "${CRM_APP}"

    export WA_SUITECRM="http://localhost:8080"

    trap 'echo "[$(date +%H:%M:%S)] Stopping Apptainer CRM instances..."; \
          apptainer instance stop "'"${CRM_APP}"'" 2>/dev/null || true; \
          apptainer instance stop "'"${CRM_MARIADB}"'" 2>/dev/null || true' EXIT

    echo "[$(date +%H:%M:%S)] Waiting for SuiteCRM at ${WA_SUITECRM} ..."
    MAX_WAIT_CRM=600
    WAITED_CRM=0
    until curl -sf "${WA_SUITECRM}" > /dev/null 2>&1; do
        sleep 10
        WAITED_CRM=$((WAITED_CRM + 10))
        if [ "${WAITED_CRM}" -ge "${MAX_WAIT_CRM}" ]; then
            echo "ERROR: SuiteCRM did not come up after ${MAX_WAIT_CRM}s"
            exit 1
        fi
    done
    echo "[$(date +%H:%M:%S)] SuiteCRM ready after ${WAITED_CRM}s"
    fi
else
    echo "[$(date +%H:%M:%S)] Using external SuiteCRM: ${WA_SUITECRM}"
fi

# ── Dry-run mode ──────────────────────────────────────────────────────────────

if [ "${DRY_RUN}" = "1" ]; then
    echo "[$(date +%H:%M:%S)] DRY RUN — skipping vLLM startup and collection"
    # shellcheck disable=SC2086
    python scripts/collect_safe_trajectories.py \
        --dry-run \
        --backend vllm \
        --vllm-url "${VLLM_URL}" \
        --model "${MODEL}" \
        --task-ids ${RESOLVED_TASKS}
    exit $?
fi

# ── Start vLLM server ─────────────────────────────────────────────────────────

echo "[$(date +%H:%M:%S)] Starting vLLM (tensor-parallel=${TP_SIZE})..."
VLLM_USE_FLASHINFER_SAMPLER=0 python -m vllm.entrypoints.openai.api_server \
    --model "${MODEL}" \
    --tensor-parallel-size "${TP_SIZE}" \
    --port "${VLLM_PORT}" \
    --max-model-len 8192 \
    --dtype bfloat16 \
    --trust-remote-code \
    > "${LOG_DIR}/vllm_${SLURM_JOB_ID:-local}.log" 2>&1 &
VLLM_PID=$!
echo "  vLLM PID: ${VLLM_PID} — log: ${LOG_DIR}/vllm_${SLURM_JOB_ID:-local}.log"

# Kill vLLM on exit (CRM instance cleanup trap already set above)
trap 'echo "[$(date +%H:%M:%S)] Caught signal — killing vLLM ${VLLM_PID}"; kill ${VLLM_PID} 2>/dev/null || true; \
      apptainer instance stop "suitecrm_${SLURM_JOB_ID:-local}" 2>/dev/null || true; \
      apptainer instance stop "mariadb_${SLURM_JOB_ID:-local}" 2>/dev/null || true' EXIT

# ── Wait for vLLM health ──────────────────────────────────────────────────────

echo "[$(date +%H:%M:%S)] Waiting for vLLM /health ..."
MAX_WAIT=900  # Qwen-72B needs ~170s to read from Lustre + shard loading time
WAITED=0
until curl -sf "${VLLM_URL%/v1}/health" > /dev/null 2>&1; do
    # Detect if vLLM crashed so we don't wait the full 900s
    if ! kill -0 "${VLLM_PID}" 2>/dev/null; then
        echo "ERROR: vLLM process (PID ${VLLM_PID}) died during startup after ${WAITED}s"
        echo "Last vLLM log lines:"
        tail -30 "${LOG_DIR}/vllm_${SLURM_JOB_ID:-local}.log" >&2
        exit 1
    fi
    sleep 5
    WAITED=$((WAITED + 5))
    if [ "${WAITED}" -ge "${MAX_WAIT}" ]; then
        echo "ERROR: vLLM did not come up after ${MAX_WAIT}s"
        echo "Last vLLM log lines:"
        tail -30 "${LOG_DIR}/vllm_${SLURM_JOB_ID:-local}.log" >&2
        exit 1
    fi
done
echo "[$(date +%H:%M:%S)] vLLM ready after ${WAITED}s"
echo ""

# ── Collect trajectories ──────────────────────────────────────────────────────

echo "[$(date +%H:%M:%S)] Starting collection (${N_RESOLVED} tasks, ${MAX_RETRIES} retries each)..."

# shellcheck disable=SC2086
python scripts/collect_safe_trajectories.py \
    --backend vllm \
    --model "${MODEL}" \
    --vllm-url "${VLLM_URL}" \
    --task-ids ${RESOLVED_TASKS} \
    --max-retries "${MAX_RETRIES}" \
    --max-steps "${MAX_STEPS}" \
    --output-dir "${OUTPUT_DIR}"

COLLECT_EXIT=$?

# ── Done ──────────────────────────────────────────────────────────────────────

echo ""
echo "[$(date +%H:%M:%S)] Collection finished (exit ${COLLECT_EXIT})"
echo "  Trajectories: ${OUTPUT_DIR}/"
ls -1 "${OUTPUT_DIR}"/*.json 2>/dev/null | wc -l | xargs printf "  JSON files saved: %s\n"

exit "${COLLECT_EXIT}"
