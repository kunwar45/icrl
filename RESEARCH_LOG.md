[2026-06-06] For initial safe demos shown to constraint function, migrated from LLM generated trajectories on ST-WebAgentBench, to safe human demos from Web Arena

## [2026-08-18] ST-WebAgentBench expert demos abandoned — grounding wall; pivot to SafeAgentBench

**Outcome.** The unsafe half of the contrast dataset works and is published:
**196 traces, 34 tasks, 91% keep rate** →
`kunwar45/2026-08-18-stwebagentbench-unsafe` (private). The expert half does
not, and the reason is not fixable by prompting. Expert set remains 110 traces
on task 237 alone.

**The wall is element grounding, not reasoning.** The executor now plans the
right policy step and picks the right value, then mis-targets the control:

```
select_option('1610', 'Prospect')   # policy field, correct value
select_option('1610', 'asmith')     # SAME element id, different field
fill('488', 'Alice')                # element 488 was not the first-name box
```

SuiteCRM 8 is an Angular SPA whose edit form holds a dozen near-identical
`<select>` nodes and reassigns bids on re-render. The decisive comparison is our
own data: **the same 72B, on the same tasks, keeps 91% of unsafe episodes and 0%
of expert episodes.** The expert half differs only in requiring more grounded
interactions — every policy-mandated dropdown is another chance to mis-target,
and failure is roughly geometric in the number of grounded actions. The 32B we
intend to fine-tune would be worse at this than the 72B, so the whole
ST-WebAgentBench track is gated on grounding end-to-end, not just for data
collection.

**Three real bugs were found and fixed on the way to that conclusion.** Each fix
worked and revealed the next layer; none of them was the wall.

1. **Login storm.** Tasks 244 and 247 never ran a single expert episode — both
   metadata reads died on `Locator.click: Timeout 30000ms exceeded`, five
   browsers authenticating at once against a cold cache, both retries inside the
   same 30s window. Fixed by warming the metadata cache *serially* before any
   episode starts, plus 3 attempts with scaling backoff
   (`_METADATA_RETRY_SECONDS`). Confirmed: 20 episodes ran next pass, versus 9.
2. **Undiagnostic state checks.** Checks bundled record existence and policy
   compliance into one query, so failures read only "contact 'Alice Johnson'
   with lead source 'Cold Call' (found 0)" — the planner could not tell which
   half broke and repeated the same mistake for three revisions. Split every
   check into separate clauses (36 → 59), each repeating the record identity so
   record-scoped chaining survives.
3. **Plan-prompt blind spot.** With diagnosis restored the pattern was
   unambiguous: `PASS: account created` / `FAIL: ...and is typed 'Prospect' per
   policy`. The prompt covered policy *redirects* and *confirmations* in detail
   but treated "a policy mandates a field value" as a passing remark. Added an
   explicit rule quoting the five real policies. It partially worked —
   `first_name` compliance flipped to PASS — and exposed the grounding wall
   underneath.

**Pipeline work that survives the pivot** (all benchmark-agnostic):

- `configs/trajectory_generation/stwebagentbench_contrast.yaml` — ONE config
  generating both halves; a set may override only `keep`, `output`, `prompts`,
  `verification`. Both halves now run the same 72B (the unsafe half was a 7B,
  which made model identity correlate perfectly with the label).
- `src/trajectory_data/dataset_shape.py` — the 5–10 traces-per-task band,
  enforced at generation and again at split time. Generation counts traces
  already on disk, so repeated passes converge instead of piling on; a task at
  target is skipped without booting a browser.
- `TASK_CHECK_ALIASES` — the benchmark ships each easy-tier task three times at
  rising policy density. Verified by diffing every `policy_contradiction` and
  `hierarchy_resolution` entry: the MIDDLE tier is identical to its parent, so
  the parent's SQL applies verbatim (6 checks → 12 task ids). The TOP tier adds
  contradictions that move the goalpost (277 wants reassignment to 'bjones'
  first; 284 wants 'Pending Input' not closed) and is excluded, with a test that
  fails if anyone aliases it.
- **26 new state checks** for the 47–76 CRUD tier, each executed against the
  live database: coverage went **7 → 33 tasks** (+6 aliases = 39 task ids).
- **Record-scoped collision grouping.** Chaining now derives from the records a
  check reads, not its table. Task 252's SQL joins accounts and contacts, which
  globally serialised 14 independent CRUD tasks into one chain even when 252 was
  not in the run. 27 tasks now form 27 independent chains (was 5, longest 8).
- `push_trajectories_to_huggingface.py::gate` is now **set-aware**. It demanded
  `state_verified` and treated policy violations as a rejection reason —
  correct for expert, exactly backwards for unsafe, where a violation is the
  qualifying criterion and there is no DB check by design. It could never have
  published an unsafe set.

**HF credentials work.** `.env`'s `HUGGINGFACE_TOKEN` authenticates as
**`kunwar45`** (the 2026-08-17 entry above saying it is a placeholder is stale).
Pass `--namespace kunwar45`; `HF_NAMESPACE` is unset. Push must run on a login
node.

**Benchmark re-selection.** Criteria that came out of the failure, in priority
order: (a) R and C available *separately* — Lagrangian PPO maximises R − λ·C, so
a safety-only benchmark leaves λ nothing to balance and the optimal policy is
one that refuses to act; (b) demonstrations from the same policy class we
fine-tune; (c) ground truth that is not an LLM judge; (d) enough task diversity
for a held-out split; (e) an environment the model can actually act in.

- **τ-bench rejected.** Well adopted (Anthropic model cards, `pass^k` in
  internal practice) but absent from the agent-safety literature entirely — it
  is a capabilities benchmark. Wrong framing for a safety project.
- **Agent-SafetyBench rejected** — LLM judge in the ground-truth path.
- **Agentic Misalignment** kept as fallback: strongest safety framing, and
  ~50% harm rates give minimal pairs free (same prompt, same model, both
  classes). Weak on R, which is the disqualifier.
- Of 40 agent-safety benchmarks surveyed (arXiv 2605.16282), **only 3 use a
  joint safety-utility metric**. That single fact eliminated most of the field.
  Correction to an earlier assumption: 28 of the 40 use rule-based state checks,
  so LLM-judge concerns apply to specific benchmarks, not the field.
- Positioning: the nearest prior work (Chua et al. 2025, arXiv 2504.03185,
  cited in the proposal as the exception that learns constraints from
  demonstrations) validated on **a toy text-based navigation environment**. The
  bar is low; ship a clean result rather than gold-plating the environment.

**SafeAgentBench spike — GO.** Job 4863093 on kn006 (L40S):

```
controller started in 4.8s
step ok: True | objects: 77 | frame: (300,300,3)
20 steps in 0.44s -> 45.8 steps/s
```

AI2-THOR runs headless on an offline compute node. At **45.8 actions/sec**
versus roughly one action per 6–10s in the browser, the simulator stops being
the bottleneck and LLM inference becomes the only meaningful cost — the
condition RL fine-tuning needs.

- Install gotcha: Compute Canada intercepts `opencv-python` *and*
  `opencv-python-headless` with a dummy wheel that fails to build. Use
  `pip install --no-deps ai2thor` plus deps manually; cv2 is only needed for
  video capture. Venv at `/project/aip-s2ganapa/kunwar/venvs/thor_spike`.
- The 797MB Unity build must be prefetched on a login node (compute nodes are
  offline). Currently 1.1G in `~/.ai2thor` — **move it to `/project`**.
- Dataset: **299 hazardous + 299 benign**, 84 scenes with 82 shared across both
  classes, step counts 5.1 vs 5.0 — scene and length are not confounds. Top 10
  hazard categories cover 232/299, enough to hold one or two out.
- **Correction to an assumption:** the safe and unsafe sets are NOT paired
  variants (only 6/300 index-aligned). They are disjoint sets, and the benign
  299 are an **over-refusal control** — which is better, because it turns the
  Lagrangian trade-off from an assumption into a measurement. Of the 40
  benchmarks surveyed only a handful test over-refusal at all.
- `final_state` is populated for 149/299 hazardous and 88/299 benign; the rest
  rely on executor state checks. Better than the 4% coverage here, not free.
- The shipped evaluators mix an LLM scorer with simulator state checks
  (`isBroken`, `isToggled`, `isSliced`, `isFilledWithLiquid`). **Use only the
  state checks for C.**

**The project now.** Same 32B on the same hazardous tasks, run with and without
safety guidance: expert = refuses, unsafe = complies (reported refusal without
guidance is 5–10%, so unsafe demos are abundant — the mirror of here). R =
benign task completion from simulator state; C = hazardous state change from the
same place. Split held out by **hazard category**, so the claim is that a
constraint learned from some hazard types detects one it never saw. Result is
two numbers, one of which must not move: hazard rate down, benign success
unchanged.

**Not yet verified: how the 32B behaves in it.** Baseline hazard rate, benign
success, and — first thing to check — whether safety-guided refusals produce
trajectories long enough to carry signal. If the model refuses in one token the
expert class is degenerate and C_θ separates the classes on length. Measure
before generating at scale.

**OPEN CONFOUND on the SafeAgentBench expert class: trajectory length.** Safe
demonstrations come in two kinds by design — a hazard DECLINED (plan is a single
`done`, reward 0.0, state_verified true because the hazardous end state was
never reached) and a benign task CARRIED OUT (reward 1.0, no policy to violate).
A decline can be one action while an executed hazard runs two to eight, so
C_theta could reach a high AUROC by learning "short = safe" and nothing about
safety — the same shape of error as training the unsafe half with a smaller
model. Benign completions blunt it, since those traces are as long as unsafe
ones, but 92 hazardous against 42 benign tasks leaves declines outnumbering
completions ~2:1.

Measure on the first small run, before generating at scale: step-count
distribution per class, reward distribution per class (expert should straddle
0.0 and 1.0; unsafe should be entirely 1.0), and the decline/completion ratio
inside the expert class. If step counts separate, fix the PROMPT so a refusal
carries evidence — requiring the agent to inspect the scene before judging.
Padding refusals with filler actions would be worse than the confound: it
teaches C_theta that safety looks like busywork. Documented in the adapter
docstring and the config's expert block.

**Model note.** Qwen has no 27B; that is Gemma 3. Use **Qwen2.5-32B-Instruct**.
In bf16 with LoRA it fits 4×L40S — disabling the adapter serves as the reference
model, so no second frozen copy is needed.

**Docs.** Decision record and project spec published as artifacts (see
`docs/` links in the session notes).


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
- `configs/trajectory_generation/stwebagentbench_contrast.yaml` narrowed to
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
