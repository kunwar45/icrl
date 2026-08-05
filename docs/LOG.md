# Research log

Append-only, **most recent first**. One entry per real result or major code change:
hypothesis → method → result → next steps, with absolute dates. Routine refactors,
chores, and doc edits get no entry.

---

## 2026-08-05 — HF org live: demo pool published to `icrl-finetuning`

**Method:** Publishing now targets the `icrl-finetuning` Hugging Face org
(`HF_ORG` in `.env`; https://huggingface.co/icrl-finetuning). The standing demo
pool was hand-published (public) via `scratch/publish_demo_pool_to_hf.py`, with
an org card via `scratch/publish_org_readme_to_hf.py`. `stage_publish` in
`run_experiment.py` now falls back to the cached `hf auth login` token when
`HUGGINGFACE_TOKEN`/`HF_TOKEN` is not in `.env`.

**Result:**
https://huggingface.co/datasets/icrl-finetuning/2026-06-04-stwebagentbench-suitecrm-demos
— 81 safe + 87 unsafe SuiteCRM trajectories (collected 2026-06-04 with
`qwen/qwen-2.5-72b-instruct`), raw WebArena traces, 2026-07-27 train/held-out
splits, task definitions. Card records the reward≈0 caveat on safe demos.
Deliberately NOT published: `icrl_local`/`icrl_smoke`/`smoke_final` checkpoints
and all `data/embeddings/*` caches — every one is a mock-env / tiny-random-gpt2
smoke fixture (checked `model_name` inside the .pt files), not a result.
Local copies of every published file were then deleted after sha256/git-sha1
verification against HF — the HF repo is now the only copy, and the code
treats HF as canonical: `src/data/hf_demo_pool.py` maps pool files to the HF
repo (`HF_DEMO_POOL_REPO` overrides) and auto-fetches any that are missing;
`run_experiment.py` calls it when resolving demo paths and before `encode`.
Fetch everything by hand with `python -m src.data.hf_demo_pool` (required on
the cluster login node before offline jobs). 117 unit/e2e tests pass.

**Next steps:** first real cluster run publishes its own `<date>-<run-name>`
repos via `--stages publish --hf-public` (per-run publish still defaults to
private).

## 2026-08-05 — Artifacts now publish to Hugging Face (`publish` stage)

**Method:** Added a final `publish` stage to `run_experiment.py` (runs by default):
per run it creates `datasets/<ns>/<YYYY-MM-DD>-<run-name>` (demos, splits, embeddings,
constraint head + metrics, CuP eval results, plots, report) and
`models/<ns>/<YYYY-MM-DD>-<run-name>-policy-lora` (the adapter), each with a card
recording experiment, date_generated, source commit, models, demo_source, config and
the exact rerun command. Private by default; mock-env runs are skipped; on offline
compute nodes it skips and prints the login-node command. Policy codified in CLAUDE.md
("Datasets, checkpoints and eval results go to Hugging Face").

**Result:** dry-run verified for skip paths and upload plan. Needs a write-scoped
`HUGGINGFACE_TOKEN` in `.env` (not yet set); optional `HF_ORG` selects the namespace.

## 2026-08-05 — Repo restructured to src/scripts/scratch layout

**Method:** Reorganised to the layout of the `teaching_claude_why_replication` reference
repo. `scripts/` foldered by pipeline stage (`demos/`, `constraint/`, `finetune/`,
`cot/`, `probe/`, `gridworld/`, `infra/`, `slurm/`); the standalone `gridworld/`
subproject folded in (`gridworld/icrl` → `src/icrl`, tests → `tests/unit|integration`,
configs → `configs/experiment/`); every script, SLURM template, and scratch file renamed
to an explicit, stage-identifying name (SLURM jobs are `<script-they-wrap>_job.sh`).
Added `CLAUDE.md` (agent guide), `scratch/` (default home for one-off/AI-generated
code), `docs/` (this log + `proposal.md`).

**Result:** All 117 unit/e2e tests + 36 `icrl` package tests pass;
`run_experiment.py --profile smoke --dry-run` produces correct commands for all 9
stages. `PYTHONPATH` is now repo root (`src.*`) + `src/` (`icrl.*`).

**Next steps:** none — structural only. Old paths (`scripts/train_constraint.py`,
`slurm/env.sh`, `gridworld/…`) are gone; see `CLAUDE.md` for the current map.

## 2026-06-06 — Safe demos switched to WebArena human traces

For initial safe demos shown to constraint function, migrated from LLM generated
trajectories on ST-WebAgentBench, to safe human demos from Web Arena.
