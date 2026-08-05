# CLAUDE.md — repo guide for agents

**AI agents: do NOT write to this file unless specifically asked to — and even when asked,
encourage human review of the exact diff. This file only stays useful if it stays curated;
unsupervised agent edits turn it to slop.**

Orientation + operating rules for this repo. Read this before touching anything.

## What this project is

**Adversarial Inverse Constraint RL (ICRL) for LLM orchestrator safety** on
[ST-WebAgentBench](https://github.com/segev-shlomov/ST-WebAgentBench). A constraint
function C_θ is learned from **safe demonstrations only**, using the adversarial
principle: *if safe demos are near-optimal, any trajectory that beats their reward must
have skipped a required safety step*. The learned constraint then drives a
Lagrangian-constrained LoRA fine-tune of the policy. Headline metric: **CuP (Completion
under Policy)** on held-out tasks, baseline vs tuned.

- `README.md` — the full cluster runbook (SLURM, Apptainer, SuiteCRM). Read it before any cluster work.
- `docs/proposal.md` — the research design and the gridworld reference implementation.
- `docs/LOG.md` — chronological research log, most recent first.

## Where code runs

- **Laptop**: `--profile smoke` (tiny-random models, mock env, minutes, no GPU) and
  `--profile local` (Qwen-0.5B, mock env). The local venv is `.venv` and **must be
  Python 3.12** — Hydra breaks on 3.14.
- **Alliance / Compute Canada cluster**: `--profile cluster` — the real experiment.
  SuiteCRM + MariaDB run as Apptainer instances on the **login node**; vLLM + Playwright
  run inside SLURM jobs on **compute nodes**. Compute nodes have **no internet**: models
  must be prefetched on the login node (`scripts/infra/prefetch_models.py`), and
  `scripts/infra/submit_experiment.sh` exports `HF_HUB_OFFLINE=1`.
- Every session starts with `source scripts/infra/session_start.sh` (module load, venv,
  PYTHONPATH, CRM reachability check). Run everything **from the repository root**.
- Imports: the repo root on `PYTHONPATH` provides `src.*`; `src/` on `PYTHONPATH`
  provides `icrl.*` (the gridworld reference package). Scripts insert the repo root
  themselves, so `python scripts/<stage>/<name>.py` works from the root.

## Where things go (keep this structure)

```
src/                    reusable, human-reviewed code (import as src.*)
  constraint/             constraint head: encoder, trainer, evaluator
  data/                   trajectories, demo collection/loaders, mock env, ST-WebAgentBench glue
  finetune/               Lagrangian PPO: lagrangian.py, dual.py, rollout.py, reward.py
  models/                 model loading
  probing/                linear probes + activation patching
  utils/                  config, logging, llm_client, compute, viz
  icrl/                   gridworld ICRL reference package (import as icrl.*)
scripts/                pipeline drivers, foldered by stage; a script does no real work —
                        it pipes src/ functions together. Top level ONLY for multi-stage pipers:
  run_experiment.py       THE driver: preflight → splits → encode → constraint → gate →
                          eval_base → finetune → eval_tuned → plots
  preflight.py            fail-fast environment check (stage 0)
  sweep_finetune_hyperparams.py   multi-seed / multi-epsilon fine-tune sweeps (local or SLURM array)
  make_experiment_plots.py        figures + self-contained report.html from a finished run
  demos/                  demo collection + splits:
                            collect_safe_trajectories.py (CuP=1 easy-tier traces, THE slurm job's driver)
                            collect_suitecrm_safe_unsafe_demos.py (sampled safe+unsafe SuiteCRM demos)
                            collect_stwebagent_demos.py / collect_webarena_demos.py (other sources)
                            make_train_eval_splits.py (stage 1), verify_trajectories.py,
                            run_single_task_demo.py (one live episode), smoke_test_demo_collection.py,
                            discover_task_ids.py
  constraint/             encode_trajectories.py (stage 2), train_constraint.py (stage 3),
                          eval_constraint.py (stage 4, AUROC gate)
  finetune/               run_finetune.py (stage 6, Lagrangian), eval_finetune.py (stages 5+7, CuP)
  cot/                    cot_build_dataset.py, cot_finetune.py
  probe/                  run_linear_probing.py, run_activation_patching.py
  gridworld/              gridworld reference entry points: run_gridworld_icrl_loop.py,
                          run_stwebagent_icrl_loop.py, generate_safe_demos_llm_policy.py
  infra/                  cluster + CRM infra: setup_cluster.sh, session_start.sh,
                          cluster_probe.sh, submit_experiment.sh, prefetch_models.py,
                          start_suitecrm_apptainer.sh (cluster), start_suitecrm_docker.sh (laptop),
                          install_suitecrm_direct.sh, setup_fork_remotes.sh
  slurm/                  sbatch templates, each named <script-it-wraps>_job.sh
                          (collect_safe_trajectories_job.sh, encode_trajectories_job.sh,
                          train_constraint_job.sh, run_finetune_job.sh, ...) + env.sh (shared env)
configs/                Hydra YAML, foldered by stage (base.yaml + compute/ constraint/ cot/
                        demos/ experiment/ finetune/ probe/). NEVER hardcode hyperparams in
                        scripts; new configs go in the folder for their stage.
scratch/                one-off and AI-generated scripts (verification snippets, probes).
                        Default home for new experimental code; NOTHING imports from it.
docs/                   LOG.md (append-only research log, most recent first) + proposal.md
tests/                  fast unit tests. Top-level test_*.py cover src.*; tests/unit/ covers
                        icrl.*; tests/integration/ runs the full ICRL loop.
data/                   demos, splits, embeddings staged for runs (bulk files gitignored)
outputs/ checkpoints/ logs/ trajectories/ embeddings/   run artifacts (gitignored)
```

**Respect the structure when adding code:**

- `src/` holds verified, reusable code a human has reviewed. Placement follows what the
  code *does* (data → `src/data/`, constraint learning → `src/constraint/`, ...).
- `scripts/` holds core pipelines we expect to rerun; each stays a thin CLI over `src/`.
  If a script grows logic worth reusing, the logic moves into `src/`. It is very rare
  that an AI-written script should land in `scripts/` without human consultation —
  **default to `scratch/`** until it earns promotion.
- A new config or script goes in the folder for the stage it belongs to — never at the
  top level of `configs/` or `scripts/` unless it pipes multiple stages together.
- **Integrate, don't tack on.** Extend the existing module rather than adding a sibling
  (`foo_v2.py`, `foo_new.py` are forbidden shapes).
- Scripts inside `scripts/<stage>/` sit one level below the root: their repo-root
  `sys.path` insert uses three `.parent`s and their Hydra decorator uses
  `config_path="../../configs"`. Keep both when adding a script to a stage folder.

**Naming conventions (follow these for every new file):**

- **Names are self-describing and explicit.** A file name states what the file does and
  which pipeline stage it belongs to, so nobody has to open it to know what it is:
  `collect_suitecrm_safe_unsafe_demos.py`, not `collect_demos.py`;
  `sweep_finetune_hyperparams.py`, not `sweep.py`; `start_suitecrm_docker.sh` vs
  `start_suitecrm_apptainer.sh`, never a bare `start_suitecrm.sh`. When in doubt, the
  longer, clearer name wins.
- Scripts are verb-first: `collect_*`, `make_*`, `train_*`, `eval_*`, `run_*`,
  `encode_*`, `verify_*`, `sweep_*`, `generate_*`.
- **SLURM templates are named `<script-they-wrap>_job.sh`** — `train_constraint_job.sh`
  wraps `scripts/constraint/train_constraint.py`. The pairing must be greppable from
  either side.
- Configs: `configs/<stage>/<subject>[_<variant>].yaml`, variants appended with
  underscores. Stage lives in the folder; the subject in the name.
- `scratch/` scripts say what kind of one-off they are: `verify_*` (contract checks),
  `manual_test_*` (hand-run smoke tests — never `test_*`, which pytest would collect).
- **Never rename a pipeline stage name** (`preflight, splits, encode, constraint, gate,
  eval_base, finetune, eval_tuned, plots`) — stage names are the shared vocabulary
  between `run_experiment.py`, the README tables, and the run reports.

## The pipeline (one driver runs every stage)

```bash
python scripts/run_experiment.py --profile smoke|local|cluster
```

Stages: `preflight, splits, encode, constraint, gate, eval_base, finetune, eval_tuned,
plots` — subset with `--stages`, inspect with `--dry-run`, extra Hydra overrides with
`--override key=value`. Per-run report in `<logs>/<run_name>_experiment.json`; figures +
`report.html` in `<logs>/<run_name>/plots/`. On the cluster:
`bash scripts/infra/submit_experiment.sh` (env vars `PROFILE`, `RUN_NAME`, `STAGES`).

- The `mock` env (`src/data/mock_env.py`) is a **test fixture, not a benchmark** — never
  report numbers from it.
- CuP is always measured on held-out tasks the policy did not train on.
- Demo quality is the experiment: what ICRL treats as near-optimal expert behaviour
  matters more than any hyperparameter. `preflight` reports trajectory counts and
  clean-termination rates per source — read them before training, and say so in results
  when the safe demos are weak (most collected safe demos have reward ≈ 0, which is only
  half of what ICRL assumes).

## Git etiquette

The user stages and commits their own changes. **Never run `git add` or `git commit`**
(and never push). Leave the working tree for the user to review.

## Secrets

All credentials live in one gitignored `.env` at the repo root (`cp .env.example .env`).
`OPENROUTER_API_KEY` (collection), `HUGGINGFACE_TOKEN` (weights), `WA_SUITECRM` (CRM URL).
Never print, log, or commit a secret value; new env vars go into `.env.example` (names
and comments only).

## Gotchas (these WILL bite you — all learned the hard way)

1. **Local venv is Python 3.12** (`.venv`). Hydra breaks on 3.14. On the cluster the venv
   is `/scratch/$USER/venvs/icrl_v4` (`module load gcc python/3.12 arrow cuda cudnn`).
2. **ST-WebAgentBench validates every site URL at import** — `.env` must set `WA_SUITECRM`
   **and also `GITLAB` and `SHOPPING_ADMIN`**, even for SuiteCRM-only tasks. Point the
   extra two at SuiteCRM; nothing in the easy tier dereferences them.
3. **Apptainer**: use `instance run` (NOT `instance start` — that only runs `appinit`),
   and the persistent ext3 overlay the start script manages (a `--writable-tmpfs` SIF
   fills up and MariaDB dies with `Read-only file system` / `No space left on device`).
   Recreate a corrupted overlay with `start_suitecrm_apptainer.sh --reset-overlay`.
4. **Compute nodes are offline.** Prefetch on the login node
   (`python scripts/infra/prefetch_models.py --profile cluster`), verify with `--check`;
   preflight fails with the exact prefetch command if a model is missing.
5. **`#SBATCH` directives are literal text** — they cannot read environment variables.
   Allocation/GPU/partition are discovered by `scripts/infra/cluster_probe.sh` and passed
   on the `sbatch` command line by `submit_experiment.sh` (`ICRL_ACCOUNT`, `ICRL_GPU`,
   `ICRL_PARTITION`).
6. **`/scratch` vs `/project`**: some allocations have no usable `/scratch`; export
   `SCRATCH` and `ICRL_ROOT` *before* sourcing `session_start.sh` or everything looks for
   the repo/venv in the wrong place.
7. **SuiteCRM must be reachable from compute nodes**: use the login-node hostname in
   `WA_SUITECRM`, never `localhost`.
8. **Rollout stages need a browser** (`eval_base`, `finetune`, `eval_tuned`); the front
   half of the pipeline (`preflight,splits,encode,constraint,gate,plots`) runs without
   SuiteCRM, Playwright, or the benchmark installed.
9. **CuP = 0.000 from a weak model is indistinguishable from a broken metric** — that is
   what `tests/test_pipeline_e2e.py` pins with scripted policies (confirm-then-delete
   scores 1, delete-immediately scores 0). Run it after touching env/eval code.

## Tests

```bash
export PYTHONPATH=.:src
pytest tests/unit/ -q                                             # icrl.* units, seconds
pytest tests/ --ignore=tests/test_reasoning_trace.py \
              --ignore=tests/integration -q                       # src.* + e2e contract
```

## When you finish a task

- Append a `docs/LOG.md` entry (most-recent-first): hypothesis → method → result → next
  steps, with absolute dates. LOG.md is for **experiments and major code changes only** —
  routine refactors, chores, and doc edits get no entry.
- Update `README.md` if you added a step or changed how to run things.
- Leave staging and committing to the user.
