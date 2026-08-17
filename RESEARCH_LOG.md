[2026-06-06] For initial safe demos shown to constraint function, migrated from LLM generated trajectories on ST-WebAgentBench, to safe human demos from Web Arena

## [2026-08-17] Safe-demo generation closed out — 110 verified traces, 1 task

**What exists now.** `expert_synthetic` = **110 traces, all task 237**
("re-stage the opportunity instead of deleting it, after one consent message").
Every trace: `state_verified: true` (the database confirmed the change AND did
not already show it before the episode), `cup: true`, zero policy violations
judged over the action sequence, 8–15 steps, 99/110 kept at plan revision 0.

- Durable copy (`/scratch` is purge-eligible):
  `/project/aip-s2ganapa/kunwar/datasets/2026-08-17-stwebagentbench-expert_synthetic/`
  — holds `expert_synthetic/` (110 + dataset card + manifest + summaries) and
  `_quarantine_false_passes/` (7 rejects).
- Working copy: `$SCRATCH/trajectories/stwebagentbench/expert_synthetic/`.
- HF push is **staged but not done**: all 110 pass
  `push_trajectories_to_huggingface.py --dry-run`; `.env`'s
  `HUGGINGFACE_TOKEN` is still the literal placeholder `your_token_here`.

**Two properties of these traces that will confuse a reader.** Both are known
and neither disqualifies a trace:

- `reward: 0.0` on all 110. The benchmark's task evaluator reads the agent's
  last page; a compliant agent that re-staged and navigated on scores 0. The
  `cup_state` keep rule takes completion from the database instead.
- One `hierarchy_adherence` entry in `page_scraped_violations` on all 110, with
  `eval_types: ['is_program_html']` — the same page-scraping false positive.
  Binding verdicts live in `safety_report`; those are empty.
  `scripts/verify_trajectories.py` predates this distinction and reports
  "0/5 passed" on a perfectly good set — it checks reward and raw violations.
  Do not use it as the gate; `push_trajectories_to_huggingface.py::gate` is the
  current, correct one.

**Runs.** Job 4841380 (4×L40S, `CYCLES=80`, `RESEED_BEFORE_RUN=1`) added 30
traces in 2h14m before being stopped deliberately at 110; job 4832448 produced
the first 80. Keep rate fell from ~89% of cycles in the earlier pass to ~76%
here, cycle time ~3.4 min. Stopping early was the right call: marginal traces on
task 237 are worth far less than a second task.

**Also done 2026-08-17.**

- The five legacy task-236 traces were moved to `_quarantine_false_passes/`.
  Its database check is sound, but its `strict_execution` policy is
  unsatisfiable here (`is_sequence_match` wants element text "ok"; this dialog
  says "Proceed"), and `SequenceEvaluator` is dormant unless the episode ends on
  `answer()` — so `violated=False` was recorded by dormancy, not compliance.
- `configs/trajectory_collection/stwebagentbench_unsafe.yaml` narrowed to
  `[237]` so the unsafe list matches the expert list, per its own rule: unequal
  lists let a constraint head separate the classes by task identity.

**The bottleneck is verifiability, not compute.** Of the 20 easy-tier SuiteCRM
tasks (235–254): 13 have no database check at all; 236's policy is
unsatisfiable; 244/246/247/248/252 have checks but the executor never saves the
record (five passes, zero verified traces). Most of the 13 are *unseeded
fixtures*, not benchmark limits — `acl_roles` empty, `securitygroups` empty, no
inbound email, no uploaded document, and for 242/243 the record to be created is
already seeded. `reseed_suitecrm_demo_data.sh` already repairs one such defect
(the demo users the seed inserts with `status=1` and no `user_hash`, which
SuiteCRM rejects).

**Next, in order.**

1. Fix the save failure for 244/246/247/248/252 — they already have database
   checks, so this is executor-side only (the per-step DB probe that fires
   "GOAL ACHIEVED" can equally fire "NOT SAVED YET"). 1 → 6 tasks.
2. Seed the missing fixtures and write one SQL check per recovered task. +~6.
3. Patch the fork's policy definitions where they are deployment bugs (236's
   "ok"/"Proceed", 240/241's whitelist that cannot express a time entered
   through a split hour/minute widget). +~3.
4. Unsafe collection over whatever list results, then the contrastive build.
5. Only after C_θ shows cross-task transfer: a second app (GitLab) for
   cross-domain generalization.

Target before phase 2 is meaningful: **≥10 tasks, ≥3 policy families,
~25 safe + ~25 unsafe each, with a held-out task split.** 110 traces of one task
is a working pipeline, not a dataset — there is no held-out task to test whether
C_θ learned the constraint or learned task 237.
