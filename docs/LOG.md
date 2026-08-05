# Research log

Append-only, **most recent first**. One entry per real result or major code change:
hypothesis → method → result → next steps, with absolute dates. Routine refactors,
chores, and doc edits get no entry.

---

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
