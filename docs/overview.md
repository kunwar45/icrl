# Project overview — what this codebase does

Constrained fine-tuning of an LLM web agent using **Inverse Constraint
Reinforcement Learning (ICRL)**. The initial target benchmark is
[ST-WebAgentBench](https://github.com/segev-shlomov/ST-WebAgentBench), where an
agent operates a real CRM (SuiteCRM) through a browser under explicit safety
policies; other environments are planned later. The method itself is
environment-agnostic — it needs only trajectories plus a ground-truth safety
verdict per episode (the `safety_report` contract), so the benchmark is a
pluggable component, not part of the method.

The core idea: nobody hand-writes a safety reward. Instead, a **constraint
function C_θ is inferred by contrast** between safe expert demonstrations and
unsafe rollouts, then a policy is fine-tuned to maximize task reward while
keeping C_θ below a budget.

## The method in one paragraph

Collect two sets of trajectories: *expert* demos (a strong model following the
policies, kept only when it succeeds with zero violations) and *unsafe* demos
(a weak model with the safety prompt removed — its violations are the signal).
Train a constraint head C_θ over trajectory embeddings so expert trajectories
score below an anchor β and unsafe/policy trajectories score above it
(`src/icrl_dual_training/constraint_trainer.py`). Verify C_θ generalizes via AUROC on held-out
tasks. Then fine-tune the policy with **Lagrangian constrained PPO** — this is
*not* RLHF: there is no preference data and no learned reward model. Task
reward R is the benchmark's own success signal (kept deliberately separate so
safety opinions don't leak into R — see `src/lagrangian_finetuning/reward_model.py`), and the
objective is

```
maximize  R(τ) − λ · C_θ(τ)
```

with λ adapted by dual ascent (`src/lagrangian_finetuning/dual_variable.py`) to keep expected
constraint cost under a budget ε. The policy is trained as a LoRA adapter.

## Pipeline stages

One driver runs everything: `python scripts/run_experiment.py --profile <p>`.

| # | Stage | Script | What it does |
|---|-------|--------|--------------|
| 0 | `preflight` | `run_preflight_checks.py` | Fail fast on a broken env; report demo-source quality (count, distinct tasks, clean terminations) |
| 1 | `splits` | `make_demo_splits.py` | Split demos into train / held-out **by task ID**, so the gate never sees training tasks |
| 2 | `encode` | `embed_trajectories.py` | Embed each trajectory (`safe.pt`, `unsafe.pt`) |
| 3 | `constraint` | `train_constraint.py` | Train C_θ: expert scores anchored below β, policy/unsafe scores pushed high |
| 4 | `gate` | `evaluate_constraint.py` | AUROC on held-out tasks — if C_θ can't separate safe from unsafe on unseen tasks, fine-tuning against it is meaningless (`--strict-gate` stops the run) |
| 5 | `eval_base` | `evaluate_policy.py` | CuP of the **untuned** policy (baseline) |
| 6 | `finetune` | `finetune_policy.py` | Lagrangian PPO with LoRA (`src/lagrangian_finetuning/lagrangian_ppo_trainer.py`) |
| 7 | `eval_tuned` | `evaluate_policy.py` | CuP of the **tuned** policy |
| 8 | `plots` | `make_experiment_plots.py` | Figures + self-contained `report.html` |

The headline metric is **CuP (Completion under Policy)**: the task succeeded
AND zero policies were violated, always measured on tasks the policy did not
train on. Eval runs *before and after* fine-tuning so the claim is "constraint
following improved without killing task performance" — a single post-hoc eval
could not show that.

## Two design decisions worth knowing

- **The constraint needs both classes.** Safe demos alone cannot define the
  boundary; C_θ is learned by contrast against unsafe rollouts. See
  [trajectory-collection.md](trajectory-collection.md) for how both sets are collected.
- **R and C are deliberately separate.** R measures *did the agent do the job*
  (benchmark ground truth, plus a small step penalty and truncation penalty);
  C measures *did it do the job safely*. An LLM judge for R would blur the
  separation the method depends on.

## Profiles

| Profile | Encoder | Policy | Env | Purpose |
|---------|---------|--------|-----|---------|
| `smoke` | tiny-random-gpt2 | tiny-random-gpt2 | mock | plumbing only — proves the wiring, says nothing about results |
| `local` | Qwen2.5-0.5B | Qwen2.5-0.5B-Instruct | mock | real weights at laptop scale |
| `cluster` | Qwen2.5-1.5B | Qwen2.5-7B-Instruct | ST-WebAgentBench | the experiment |

The policy is 7B (not the 72B used for demo collection) because it must fit a
LoRA fine-tune plus its own rollouts on the job's GPUs. `mock`
(`src/environments/mock_environment.py`) is a deterministic text CRM emitting the same
`safety_report` contract as the benchmark — a test fixture, never a source of
reportable numbers.

## Layout

```
scripts/run_experiment.py     the driver (all stages)
src/icrl_dual_training/               trajectory encoder + C_θ trainer
src/lagrangian_finetuning/                 reward model, Lagrangian PPO, dual ascent, rollouts
src/trajectory_data/                     Trajectory dataclass, trace loaders, mock env
scripts/slurm/                        cluster job templates (collection, training, sweep)
src/gridworld/                    gridworld ICRL reference implementation
```

Cluster setup, SuiteCRM/Apptainer, and SLURM specifics are covered in the
top-level [README](../README.md).
