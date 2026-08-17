# Trajectory throughput — where the wall-clock goes and what removes it

Both trajectory pipelines roll out real browser episodes against one SuiteCRM,
so they share one performance profile and one set of levers. This page is the
reference for all of it; [collection](trajectory-collection.md) and
[generation](trajectory-generation.md) describe *what* each pipeline produces.

## The shape of the cost — measured, not guessed

Measured on klogin03, 2026-08-15, task 236, lean observation on
(`scratch/probe_step_breakdown.py`). Three steps, totals per step:

| Phase | Per step | Share |
|---|---|---|
| **`_task_validate`** — the benchmark re-running its task + policy evaluators | **4.69 s** | **71 %** |
| `_get_obs` total (all extraction) | 0.55 s | 8 % |
| — of which `extract_dom_snapshot` | 0.22 s | |
| — of which `extract_merged_axtree` (required) | 0.12 s | |
| — of which `extract_screenshot` | 0.10 s | |
| — of which `_pre_extract` (required) | 0.08 s | |
| everything else (waits, page checks, action dispatch) | ~1.4 s | 21 % |
| **observed step total** | **~6.6 s** | |

**This corrects an earlier version of this page.** It claimed observation
extraction dominated and that trimming it would roughly halve step time.
Measurement says otherwise: the trimmed extractions were ~0.33 s of a 6.6 s
step, and lean observation measured a **7 % improvement** (13.25 s → 12.4 s over
two steps), not 2×. The estimate was reasoned from what the code does per step
and was simply wrong about magnitude; the profiler was the first thing that
actually asked the machine.

The real bottleneck is `validate()` — see item 0 below, which is where the
actual speedup came from.

Two consequences still drive the decisions below, both unaffected:

1. **The GPU is idle most of the time.** One episode at a time means the vLLM
   server runs at batch size 1 while the browser works — and with a 6.6 s step
   against a ~1.5 s model call, that idleness is worse than first assumed, not
   better. Concurrency is close to free throughput.
2. **The environment, not the GPU, is the scarce resource.** A pass that proves
   completion from the database needs that database to itself.

## What was removed

### 0. The evaluator re-running on every step — 4.27× measured

`src/environments/browsergym_deferred_validation.py` gates `_task_validate` to a
cheap no-op during the episode; `episode_runner` asks for one real evaluation
when the episode ends. Enabled with `episode.defer_validation: true`.

This is sound because `task.validate(page, chat_messages, trajectory)` is
**stateless per call** — it takes the whole trajectory and recomputes from
scratch, so the verdict after step N is simply superseded by the one after step
N+1, and only the last is ever read (`episode_runner` takes its safety report
from the final step; `cup_state` reads completion from the database, not from
the benchmark's reward).

Verified on klogin03, 2026-08-15 (`scratch/probe_deferred_validation.py`): the
same fixed action sequence run both ways gave **identical policy verdicts across
all 7 policies and identical reward**, at **6.28 s → 1.47 s per step (4.27×)**.

Two responsibilities move to the caller, which is why it is opt-in:

- **Termination.** `done` comes out of `validate()` — it is where the benchmark
  notices `answer()`. Gated, it never fires, so `episode_runner` detects the
  answer action itself.
- **Per-step rewards.** `reward_best` across steps is no longer observable, only
  the final score. Irrelevant to `cup_state`; a run scoring from the benchmark's
  own reward must leave this off.

Incidental fix: `validate()` appends a synthetic "stopped, too many steps"
action to the trajectory whenever `len(trajectory) >= max_steps`. Called every
step, it kept appending — the trajectory grew per call once the cap was hit.
One call cannot do that.

### 1. Observation extraction nobody reads (worth ~7 %, not the 2× first claimed)

BrowserGym's `_get_obs` runs seven extractions per step. This project reads
`axtree_object`, `goal`, `url`, `chat_messages` and the benchmark's `policies`,
and flattens the axtree with no `extra_properties` argument — so the DOM
snapshot, the per-node visibility/clickability walk, the screenshot and the
focused-element lookup are dead work on every step of every episode.

`src/environments/browsergym_lean_observation.py` stubs those extraction
functions in the namespace where the env's own `_get_obs` resolves them, leaving
`_get_obs` itself untouched. Enabled by `benchmark.lean_observation` (default
on).

It is written that way because the first version wasn't, and broke on the
cluster: it replaced `_get_obs` with a copy of BrowserGym's, and
**ST-WebAgentBench does not use BrowserGym's env class** —
`stwebagentbench/browser_env/custom_env.py` defines its own with a different
observation dict (no `goal_object`; it derives `goal` from the chat and adds
`policies` and `read_page`). Patching the extractors instead means whatever keys
an env builds, it keeps building. The fork's `read_webpage_content` — a
`wait_for_load_state('networkidle')` plus `innerText` on every step — is stubbed
the same way; nothing in the fork or in `src/` reads `read_page`,
`dom_object` or `extra_element_properties` (checked 2026-08-15).

### 2. Serial episodes

`episode.concurrency` (collection) and `generation_loop.concurrency`
(generation) run several episodes at once. Two things had to be true first:

- **BrowserGym's Playwright instance is process-global**, and Playwright's sync
  API cannot be used across threads. `src/environments/playwright_thread_isolation.py`
  makes the accessor thread-local, so each worker owns its own driver. Note it
  patches `browsergym.core.env._get_global_playwright` as well as the package's
  — `env.py` imported the function by value, so patching the package alone
  installs a patch that does nothing.
- **Workers must release what they hold.** Each worker thread owns a Playwright
  driver subprocess and a pooled database connection;
  `src/trajectory_collection/episode_concurrency.py` uses a hand-rolled pool
  rather than `ThreadPoolExecutor` precisely because the latter has an
  `initializer` but no finalizer, and a leaked driver per worker is how a long
  run exhausts the node.

**Safety comes from chaining, not from a low concurrency number.** The unit of
scheduling is a *chain*: items in one chain never overlap, chains run in
parallel. Tasks whose ground-truth checks read the same tables go in one chain,
so one episode's writes can never land inside another's before/after comparison.

The grouping is derived from the checks' own SQL
(`stwebagentbench_state_verifier.task_collision_group`), so it cannot drift away
from what the checks actually query. For the current expert task list:

```
236, 246 → leads            237, 247 → opportunities      244 → cases
248 → emails                252 → accounts+contacts
```

Seven tasks, five chains, longest chain two — so five is the most concurrency
that task list can use.

Collection chains by the same groups, and **not** because its keep rule reads
the database (`unsafe_binding` reads the trajectory only). It is about what the
demonstrations show: eight rollouts racing to delete the same lead produce one
delete and seven episodes flailing at a record that vanished mid-episode, and
`unsafe_binding` clears its substantive-action bar on flailing. That would fill
the set that *defines* unsafe behaviour with traces of an agent confused by
missing data. Rollouts of one task therefore run in sequence; different tables
overlap.

### 3. A fresh database handshake per step

`verify_task_state` runs on *every* step to drive the "goal achieved" hint, and
opened a new connection each time. It is now pooled per thread.

The connection is **autocommit**, and that is load-bearing rather than
stylistic: under MySQL's default `REPEATABLE READ` the first `SELECT` pins a
snapshot, so a reused connection would keep reporting the state as of the
episode's first step and never notice the agent's work landing — the exact
signal the probe exists to provide.

### 4. Prompt field order that defeated prefix caching

vLLM runs with `--enable-prefix-caching`, which reuses the KV cache for the
longest *shared prefix* between requests. The prompts previously put the
volatile axtree before the static action-space block, and the per-step hint at
the end of the *system* message — which invalidated every token after it, i.e.
the whole prompt, on any step where a hint fired.

Both configs now order fields static → append-only → volatile:

```
system:  instructions, plan, policies, mechanics, rules, {action_space}
user:    {goal}, {actions_so_far}, {chat_history}, {url}, {axtree}, {hint_block}
```

`{actions_so_far}` grows by one line per step, so step N+1's prompt is a strict
extension of step N's. **Reordering these fields re-introduces a full prefill on
every step.**

### 5. Passes that could not overlap

`scripts/start_suitecrm_shards.sh` starts K independent SuiteCRM+MariaDB stacks,
each with its own instances, data directory, sandbox, HTTP port, MariaDB port
and env file. A job joins one with `ICRL_SUITECRM_SHARD=n`; the lock in
`scripts/slurm/job_environment.sh` is per shard, so passes on different shards
never exclude each other.

Read the cost notes in that script before picking K: each shard needs its own
multi-GB writable sandbox, they run on the login node, and compute nodes cannot
host them today (`apptainer instance start` fails there with a rootless-cgroup
dbus error — see `logs/slurm/icrl-gen_45204272.err`). 2–4 is the useful range.

### 6. Smaller things

- `slow_mo_ms` 250 → 150. Playwright applies it per *operation*, and one agent
  action is several operations.
- `RESEED_TABLES=leads,opportunities` restores only what a cycle consumed.
  Names expand to their related tables (an email lives in `emails` plus three
  relationship tables) because a partial restore that strands relationship rows
  is worse than no optimisation — the drift is invisible until a check starts
  failing for no visible reason. Unset means the full, always-correct reseed.
- `collect_trajectories_job.sh` now takes the **same** SuiteCRM lock as the
  generation wrapper, and gained `CYCLES` / `RESEED_BEFORE_RUN`. It previously
  took no lock, so an unsafe collection pass could run underneath a generation
  pass and change the database inside its before/after comparison — poisoning
  expert traces that then looked verified and proved nothing.

## What was considered and rejected

**Reusing the browser across revisions.** BrowserGym's `reset()` explicitly
closes the context, chat and browser and relaunches Chromium
(`browsergym/core/env.py`), and re-runs the task's SuiteCRM login. Holding the
env object across revisions therefore saves only the `gym.make` call. The cost
is real but it is not reachable from this side of the seam.

## Running 150 + 150

```bash
# Login node, once: shards up, wizard completed per shard, models prefetched
bash scripts/start_suitecrm_shards.sh --shards 2
python scripts/prefetch_models.py --profile cluster

# Expert — several reseed-and-generate cycles per allocation, one job per shard
ICRL_SUITECRM_SHARD=1 CYCLES=10 RESEED_BEFORE_RUN=1 \
  CONFIG=configs/trajectory_generation/stwebagentbench_expert.yaml \
  sbatch --account=$ICRL_ACCOUNT --gres=gpu:h100:4 scripts/slurm/generate_trajectories_job.sh

# Unsafe — cheap; reseed between cycles so late rollouts still act on real records
ICRL_SUITECRM_SHARD=2 CYCLES=4 RESEED_BEFORE_RUN=1 \
  CONFIG=configs/trajectory_collection/stwebagentbench_unsafe.yaml \
  sbatch --account=$ICRL_ACCOUNT --gres=gpu:h100:1 scripts/slurm/collect_trajectories_job.sh
```

A generation cycle yields **at most one trace per task**: the keep rule is
differential (`state_verified = persisted_after and not state_satisfied_before`),
so once a task's goal state is reached, no later episode in that cycle can be
credited with reaching it. Traces per cycle is therefore bounded by the number
of tasks that keep, which is why the expert task list matters more than any
speed lever — and why `TASK_STATE_CHECKS` gaining a task is worth more than
another shard.

## Before trusting a number from this page

Every figure here is derived from what the code does per step, not from a
measured cluster run. The first real pass after these changes should be checked
against `summary.csv` and the job's timestamps, and this page corrected.
