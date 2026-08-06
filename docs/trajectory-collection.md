# Data collection — expert and unsafe trajectories

ICRL never gets told what "unsafe" means. It infers the constraint function C_θ
by contrasting two trajectory sets, so **the collected data defines safety for
the entire experiment** — what goes in here matters more than any
hyperparameter downstream.

*Collection* means rolling out real agents in the real benchmark environment
and keeping the episodes the benchmark's **ground-truth evaluators** label —
never a model's opinion of itself. The parallel synthetic route —
plan-guided [trajectory generation](trajectory-generation.md) — shares this
pipeline's adapter seam, trace schema, and verification rule.

## Architecture: one engine, config-defined runs

```
scripts/collect_trajectories.py            THE entrypoint (thin CLI)
configs/trajectory_collection/<run>.yaml         fully defines a run: benchmark, tasks,
                                           model, prompt, episode mode, keep rule
src/trajectory_collection/
  collection_runner.py                     task loop, keep rules, saving, manifest
  episode_runner.py                        one episode: prompt → LLM → action → step
  benchmark_adapter.py                     the benchmark-agnostic seam (+ registry)
  stwebagentbench_adapter.py               everything ST-WebAgentBench-specific
scripts/slurm/collect_trajectories_job.sh          the ONLY way runs execute for real
```

Adding a **variant** (new model, prompt ablation, different keep rule) = one new
config. Adding a **benchmark** = one new `<benchmark>_adapter.py` plus configs.
Never a new script.

## The two runs

| Config | Model | Prompt | Episode mode | Keep when |
|---|---|---|---|---|
| `stwebagentbench_expert.yaml` | Qwen2.5-72B (strong) | full policy block + safety rules | `retry_until_kept`, ≤10 retries, temperature 0.0 then ramping | `cup_one`: reward=1.0 ∧ zero violations ∧ terminated cleanly |
| `stwebagentbench_unsafe.yaml` | Qwen2.5-7B (weak) | **no policies, no safety rules** | `fixed_rollouts`, 5 per task, temperature 0.8 | `any_violation`: ≥1 ground-truth violation |

The asymmetry is deliberate. Expert data is expensive (strong model, retries,
strict filter); unsafe data is cheap — a weak model that was never told the
rules skips confirmations on its own, and its incompetence is the signal. Both
runs cover the same task set, since C_θ is learned by contrast.

The `cup_one` rule requires clean termination, not just CuP: "safe because it
dithered until the step limit" is not expert behaviour (this exact failure made
79/81 of the first-ever safe batch unusable).

## Running it (Compute Canada SLURM only)

```bash
# Login node, once per session: SuiteCRM up + model prefetched
bash scripts/start_suitecrm_apptainer.sh
python scripts/prefetch_models.py --profile cluster

# Expert (GPU count must match the config's model.tensor_parallel):
CONFIG=configs/trajectory_collection/stwebagentbench_expert.yaml \
  sbatch --account=$ICRL_ACCOUNT --gres=gpu:h100:4 scripts/slurm/collect_trajectories_job.sh

# Unsafe:
CONFIG=configs/trajectory_collection/stwebagentbench_unsafe.yaml \
  sbatch --account=$ICRL_ACCOUNT --gres=gpu:h100:1 scripts/slurm/collect_trajectories_job.sh

# Subset / overrides without editing the config:
OVERRIDES="benchmark.task_ids=[235,236]" CONFIG=... sbatch ...

# Wiring check, no GPU/browser:
DRY_RUN=1 sbatch --gres= --mem=4G --time=00:10:00 scripts/slurm/collect_trajectories_job.sh
```

The wrapper reads `model.{backend,name,tensor_parallel}` out of the config,
starts vLLM accordingly, waits for health, runs the collector, and exits
non-zero if nothing was kept. Compute nodes are offline (`HF_HUB_OFFLINE=1`);
preflight-style failures happen at startup, not after 11 hours.

On a laptop, only smoke runs are allowed (OpenRouter backend, 2 tasks, output
under `data/smoke/` so it can never be mistaken for real data):

```bash
python scripts/collect_trajectories.py \
  --config configs/trajectory_collection/stwebagentbench_expert.yaml --smoke
```

## What a collected trace looks like

One JSON per kept episode, `task_<id>_trace_<n>.json`, next to a `manifest.json`
(the resolved config + counts — every trace directory is self-describing) and
`summary.csv` (per-task attempts, kept, violation categories seen):

```json
{
  "task_id": 235,
  "collection": "stwebagentbench_expert",
  "benchmark": "stwebagentbench",
  "set": "expert",
  "model": "Qwen/Qwen2.5-72B-Instruct",
  "reward": 1.0,
  "cup": true,
  "terminated": true,
  "policies": [ ... ],          // full org/user/task policy text
  "safety_report": [ ... ],     // every evaluator dimension, violated or not
  "steps": [ {"step_idx": 0, "action": "click('30')", "observation": "<axtree>"} ]
}
```

Output root: `$SCRATCH/trajectories/<benchmark>/<set>/`. Downstream,
`src/trajectory_data/demo_loader.py` converts traces to the `Trajectory` dataclass, and
`run_experiment.py` auto-discovers these directories when `--safe-demos` /
`--unsafe-demos` are not given.

## Data hygiene

Collected trajectories stay on-cluster and **unpublished until verified**
training-worthy (`preflight` / `scripts/verify_trajectories.py`: count, distinct
tasks, clean terminations per source). Only a verified-clean dataset is pushed
to HF, dated: `<namespace>/<YYYY-MM-DD>-<benchmark>-<set>`. Everything
collected before 2026-08-05 was deleted rather than kept — an unusable demo set
that lingers is worse than an empty directory, because downstream stages will
silently train on it.
