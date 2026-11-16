# Project overview — what this codebase does

Constrained fine-tuning of LLM agents with **Inverse Constrained Reinforcement
Learning (ICRL)**: infer a constraint nobody wrote down from expert
demonstrations, then fine-tune the policy to maximise task reward subject to
that learned constraint. Benchmarks are pluggable (adapters + configs); the
current one is **secret-keeping** (a chatbot holds a secret code in its system
prompt and must not reveal it, while an LLM judge rewards doing what the user
asks). Earlier tracks: ST-WebAgentBench (its pipeline is described below), ODCV, MACHIAVELLI.

## The method (the standard every new run follows)

ICRL as defined by Malik, Anwar, Aghasi & Ahmed, *Inverse Constrained
Reinforcement Learning*, ICML 2021 (arXiv:2011.09999), and by the survey Liu et
al. 2024 (arXiv:2409.07569, Sec. 4.3 / 6.2): **alternate a constraint update and
a constrained policy update until the policy reproduces the expert.** Each round:

1. **Sample the current policy** on the training contexts (no filtering).
2. **Inverse step.** Update a feasibility function φ_ω(x, y) ∈ (0, 1)
   (frozen LLM encoder + MLP head; cost c = 1 − φ) by the maximum-entropy
   likelihood of the expert data: raise φ on expert actions, lower it on the
   *current policy's* actions, plus Malik's sparsity regulariser toward feasible
   (`src/icrl_dual_training/maxent_constraint.py::maxent_constraint_update`).
   Warm-start from the previous round.
3. **Forward step.** Constrained RL with the Lagrangian
   `max E[r] − λ (E[c] − ε)`, implemented with **GRPO** (Shao et al. 2024,
   arXiv:2402.03300) via Hugging Face TRL on a LoRA policy, resumed across rounds
   (`src/lagrangian_finetuning/secret_keeping_constrained_grpo.py`).
4. **Dual step.** `λ ← clip(λ + η (mean cost − ε), 0, λ_max)` every GRPO step.

The ground-truth constraint check is used only to grade and to audit
(AUROC of the learned cost on each round's policy samples); it never enters the
reward or the constraint objective.

**Implemented for secret-keeping** as the `icrl_loop` stage of
`scripts/run_secret_keeping_icrl_pipeline.py`, config
`configs/lagrangian_finetuning/secret_keeping_icrl_loop_qwen7b.yaml`
(4 rounds × 150 GRPO steps, ε = 0.1).

## Baselines (not ICRL — name them as what they are)

Same pipeline script, same test set, one config each:

| Name | What it is | Config |
|---|---|---|
| Learned constraint classifier + RLHF | a supervised binary classifier (expert reply → 0, base-model reply that beats the expert's reward → 1) trained once, then GRPO with a fixed penalty λ = 5 | `secret_keeping_constrained_grpo_qwen7b.yaml` |
| SFT on expert replies | behaviour cloning on the expert demonstrations | `secret_keeping_expert_sft_qwen7b.yaml` |
| RL + exact constraint check | GRPO with the ground-truth check as the cost (upper bound for the cost signal) | `secret_keeping_grpo_exact_leak_cost_qwen7b.yaml` |
| RL + prompted LLM judge | GRPO with a hand-written YES/NO safety prompt as the cost | `secret_keeping_grpo_llm_judge_cost_qwen7b.yaml` |
| Classifier + DPO | the classifier above with Lagrangian DPO (superseded) | `secret_keeping_constrained_dpo_qwen7b.yaml` |

The one-shot classifier only borrows ICRL's *data* idea (the expert gives up
reward to avoid the constraint); without the loop against the current policy
and the max-entropy objective it is not ICRL, and a policy trained long enough
against it learns to slip past it.

## Earlier ST-WebAgentBench pipeline (one-shot constraint + Lagrangian PPO)

This predates the standard above: C_θ is trained once on a fixed expert/unsafe
contrast set, then frozen for Lagrangian PPO. Kept for reference.

Collect two sets of trajectories: *expert* demos (a strong model following the
policies, kept only when it succeeds with zero violations) and *unsafe* demos
(a weak model with the safety prompt removed — its violations are the signal).
Train a constraint head C_θ over trajectory embeddings so expert trajectories
score below an anchor β and unsafe/policy trajectories score above it
(`src/icrl_dual_training/constraint_trainer.py`). Verify C_θ generalizes via AUROC on held-out
tasks. Then fine-tune the policy with **Lagrangian constrained PPO**. Task
reward R is the benchmark's own success signal (kept deliberately separate so
safety opinions don't leak into R — see `src/lagrangian_finetuning/reward_model.py`), and the
objective is

```
maximize  R(τ) − λ · C_θ(τ)
```

with λ adapted by dual ascent (`src/lagrangian_finetuning/dual_variable.py`) to keep expected
constraint cost under a budget ε. The policy is trained as a LoRA adapter.

### Pipeline stages (ST-WebAgentBench driver)

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

### Two design decisions worth knowing

- **The constraint needs both classes.** Safe demos alone cannot define the
  boundary; C_θ is learned by contrast against unsafe rollouts. See
  [trajectory-collection.md](trajectory-collection.md) for how both sets are collected.
- **R and C are deliberately separate.** R measures *did the agent do the job*
  (benchmark ground truth, plus a small step penalty and truncation penalty);
  C measures *did it do the job safely*. An LLM judge for R would blur the
  separation the method depends on.

### Profiles

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
src/icrl_dual_training/               trajectory encoder, max-entropy ICRL constraint (maxent_constraint.py), one-shot C_θ trainers
src/lagrangian_finetuning/                 GRPO forward step with dual ascent, reward models, Lagrangian PPO/DPO (earlier), rollouts
scripts/run_secret_keeping_icrl_pipeline.py  secret-keeping: demos, base eval, icrl_loop (the method), baselines, evaluation
src/trajectory_data/                     Trajectory dataclass, trace loaders, mock env
scripts/slurm/                        cluster job templates (collection, training, sweep)
```

Cluster setup, SuiteCRM/Apptainer, and SLURM specifics are covered in the
top-level [README](../README.md).
