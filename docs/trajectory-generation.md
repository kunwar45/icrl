# Trajectory generation — synthetic expert trajectories

The parallel alternative to [data collection](data-collection.md): instead of
hoping a strong model stumbles into CuP=1 rollouts, **synthesize what an
optimal trajectory should be, execute it, and let the benchmark's ground-truth
evaluators verify it**. This is synthetic data generation for SFT: a staged
draft → refine → generate → verify pipeline, adapted to embodied browser tasks.

## The stages

| Stage | What happens | Who |
|---|---|---|
| Task intake | Take a benchmark task (goal + policies) | `adapter.task_metadata` |
| **propose_plan** | What would an optimal, policy-compliant trajectory look like? Intent-level steps, not clicks | planner model |
| **refine_plan** | Critic pass: compliance, optimality, clean termination, groundability | planner model |
| **execute** | The executor model grounds the plan step-by-step in the *real* env | executor model |
| **verify + revise** (critical) | The env run is scored by the ground-truth evaluators; on failure the planner sees a concrete failure report and revises the plan, then re-execute | env + planner |

Two deliberate differences from pure text pipelines:

1. **Plans are intent-level, never action calls.** BrowserGym element ids only
   exist at runtime, so a pre-written action sequence can't be replayed. The
   executor grounds each plan step against the live page. (Pointing the
   executor at the policy model we fine-tune makes the data in-distribution
   for SFT — one config override.)
2. **Verification is execution.** A trajectory is never trusted because a model
   wrote it; it is kept only when the real rollout scores reward = 1.0, zero
   violations, and clean termination (`cup_one`). "Find a way to verify" =
   there is no judge model anywhere in the accept path.

## Architecture

```
scripts/generate_trajectories.py                      THE entrypoint (thin CLI)
configs/trajectory_generation/<run>.yaml              fully defines a run, prompts included
src/trajectory_generation/
  plan_generator.py                                   propose / refine / revise + failure report
  generation_runner.py                                per-task loop: plan → execute → verify → revise
slurm/generate_trajectories_job.sh                    starts 1–2 vLLM servers, then generates
```

Generality comes from reusing the **same `BenchmarkAdapter` seam** as
collection (`src/data_collection/benchmark_adapter.py`): a new eval needs one
adapter (which both pipelines then share) plus configs. The plan-guided episode
reuses `episode_runner.run_episode` — the plan is injected as an extra prompt
field, so all the episode robustness (loop hints, duplicate-message blocking)
carries over.

## Models: two roles, both on the job's GPUs

vLLM only — never OpenRouter for real runs. Default is **Qwen-72B for
everything**: the planner role writes, refines, and revises plans (planning
must come from a strong model, ≥27B), and the executor role grounds them —
both served by ONE vLLM server, so the job needs 4 GPUs. To make the data
in-distribution for the 7B policy instead, point the executor at a second
server; the wrapper splits `CUDA_VISIBLE_DEVICES` and starts both.

```bash
# Default (one shared 72B server, 4 GPUs):
CONFIG=configs/trajectory_generation/stwebagentbench_expert.yaml \
  sbatch --account=$ICRL_ACCOUNT --gres=gpu:h100:4 slurm/generate_trajectories_job.sh

# In-distribution 7B executor instead (5 GPUs, two servers):
OVERRIDES="models.executor.name=Qwen/Qwen2.5-7B-Instruct models.executor.vllm_url=http://localhost:8001/v1 models.executor.tensor_parallel=1" \
  CONFIG=... sbatch --account=$ICRL_ACCOUNT --gres=gpu:h100:5 slurm/generate_trajectories_job.sh

# Laptop smoke (OpenRouter backends, 1 task, output under data/smoke/):
python scripts/generate_trajectories.py \
  --config configs/trajectory_generation/stwebagentbench_expert.yaml --smoke
```

## Plan diversity

Templated plans are the pipeline's data-quality risk: a constraint head
trained on stylistically uniform experts can learn "plan-shaped = safe". Two
mechanisms, both config-driven (`diversity:` block):

1. The refine stage sees the last N **verified-successful** plans from other
   tasks (failed plans never enter the context) with an instruction to differ
   in phrasing, step granularity, and route where the task allows — never at
   the cost of compliance or optimality.
2. After refinement, the plan's character-level similarity against those
   successful plans is measured (`plan_generator.plan_similarity`); above
   `max_similarity` (default 0.85) one explicit `diversify_plan` pass runs.
   The per-task max similarity lands in `summary.csv`, so homogeneity is
   visible even when the gate never fires.

Each task gets at most `1 + generation_loop.max_plan_revisions` real-env
executions (default 4). The job exits non-zero if nothing survived
verification.

## Output

`$SCRATCH/trajectories/<benchmark>/expert_synthetic/` — the **collection trace
schema plus provenance**, so `trace_loader.py` and every downstream stage
consume it unchanged, while the source stays distinguishable (never silently
mixed with collected expert traces):

```json
{
  "set": "expert_synthetic",
  "pipeline": "trajectory_generation",
  "model": "Qwen/Qwen2.5-72B-Instruct",
  "planner_model": "Qwen/Qwen2.5-72B-Instruct",
  "plan": "1. Open the record ...",
  "plan_revisions": 1,
  "reward": 1.0, "cup": true, "terminated": true,
  "steps": [ ... ]
}
```

`manifest.json` (resolved config) and `summary.csv` (per task: kept, episodes,
revisions, outcome) sit alongside, as in collection. The same data policy
applies: on-cluster until verified, then a dated HF dataset.

## Why this can beat plain collection

Plain collection failed because the 72B *acting step-by-step* dithers: 79/81
episodes hit the step limit. Here the 72B first solves the task **once, in
text, at the level of intent** — a much easier problem — then executes with
the plan as a rail, and the failure loop feeds real evaluator verdicts back
into the plan. If a task's plan can't pass in 4 attempts, the summary says
exactly which policy fired or where it got stuck, per revision.
