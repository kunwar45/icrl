# Adversarial Inverse Constraint RL (ICRL)

Learning safety constraints for LLM orchestrators from safe-only demonstrations.

---

## The Core Idea

An LLM orchestrator that completes tasks by calling tools and prompting other agents can do so safely or unsafely. A safe agent asks for confirmation before deleting a record; an unsafe agent just deletes it. A safe agent verifies before sending an email; an unsafe agent skips that step.

The safety rules are often **implicit** — hard to write down in advance, but recognisable in good human demonstrations.

This project implements **Paper #3** of the research proposal: learning these constraints from *safe demonstrations only*, with no unsafe examples required. The trick is an adversarial principle:

> **If safe demonstrations are near-optimal, any trajectory achieving higher reward must have skipped a required safety step to get there.**

A safe agent confirms before acting (costs one step, reduces reward). An unsafe agent skips confirmation (saves a step, earns higher reward). So *higher reward than the safe demos implies unsafe behaviour*. This turns any learner trajectory that beats the safe-demo reward into an implicit negative example — no unsafe labels needed.

---

## Concepts

### Trajectory
A complete episode: a sequence of (observation, action, reward) transitions. The constraint function scores a full trajectory, not individual steps, because safety is a sequence-level property — "confirmation was never given before data access" requires seeing both steps.

### Safe Demonstrations
Human or oracle agent trajectories that complete tasks correctly, following all safety policies. Their maximum total reward defines the **adversarial threshold**.

### Adversarial Threshold
`max_reward(safe_demos)`. Any learner trajectory exceeding this is flagged as implicitly unsafe. The learner can only beat the safe demos by taking shortcuts — which are exactly the unsafe behaviours we want to learn to penalise.

### Constraint Function C_θ(τ)
A learned function mapping a trajectory to a **feasibility score** in [0, 1]. High = safe, low = unsafe. Implemented as an MLP on top of a trajectory embedding. Trained contrastively: safe demos as positives, adversarially-detected unsafe trajectories as negatives.

### PPO-Lag (Constrained Policy)
PPO augmented with a Lagrange multiplier λ. The episode cost from the constraint feeds into the Lagrangian:
```
combined_advantage = adv_reward − λ · adv_cost
λ += lr · (mean_episode_cost − cost_limit)
```
As λ rises, the policy is increasingly penalised for trajectories the constraint flags as infeasible.

---

## The Training Loop

```
Given: safe demonstrations D_safe  (near-optimal under true constraints)
       threshold = max total_reward(D_safe)

pretrain phase (optional):
  run unconstrained PPO so the policy learns to complete the task at all

ICRL loop:
  for each iteration:
    1. ROLLOUT   — run policy in environment for N steps → list of trajectories
    2. DETECT    — flag any trajectory with reward > threshold as unsafe
                   add flagged trajectories to unsafe_buffer
    3. CONSTRAIN — if enough unsafe examples accumulated:
                   train C_θ on (safe_demos, unsafe_buffer) with contrastive loss
    4. UPDATE    — run PPO-Lag update using C_θ cost as constraint signal
    5. LOG       — record metrics: reward, unsafe_buffer_size, constraint_loss,
                   feasibility_gap, λ
```

**What convergence looks like:**

| Phase | `mean_reward` | `unsafe_buffer_size` | `constraint_loss` | `λ` |
|---|---|---|---|---|
| Pretrain | rising → task solved | 0 (no detection) | — | — |
| Early ICRL | ~unsafe level (high) | growing | high, dropping | rising |
| Mid ICRL | mixed | stabilising | low | stable |
| Converged | ~safe demo level | stable | low | stable |

---

## Project Structure

The `icrl` package lives at `src/icrl/`; its runnable entry points at
`scripts/gridworld/`; its configs at `configs/experiment/`; its tests at
`tests/unit/` and `tests/integration/`.

```
icrl/  (repo)
├── configs/
│   └── experiment/
│       └── gridworld_smoke.yaml          # experiment config (start here)
├── scripts/gridworld/
│   └── run_gridworld_icrl_loop.py                  # runnable experiment script
├── tests/
│   ├── unit/
│   │   ├── test_adversarial_detector.py
│   │   └── test_constraint_loss.py
│   └── integration/
│       └── test_icrl_loop_gridworld.py
└── src/icrl/
    ├── core/
    │   ├── types.py        ← shared data structures
    │   └── interfaces.py   ← abstract base classes (swap any component)
    ├── envs/
    │   └── gridworld.py    ← toy environment
    ├── demos/
    │   ├── loader.py        ← BaseDemoLoader ABC
    │   └── gridworld_demos.py
    ├── embedders/
    │   └── mean_pool.py    ← token embedding + mean-pool
    ├── constraints/
    │   ├── losses.py        ← contrastive, margin, InfoNCE
    │   └── mlp_constraint.py
    ├── policies/
    │   ├── ppo_lagrangian.py
    │   └── rule_based.py   ← safe/unsafe oracles for testing
    ├── trainer/
    │   ├── adversarial_detector.py   ← THE adversarial principle
    │   ├── rollout_buffer.py
    │   └── icrl_trainer.py           ← main loop
    ├── metrics/
    │   └── tracker.py      ← JSON-lines + optional TensorBoard
    └── utils/
        ├── config.py        ← YAML loader
        └── seeding.py
```

---

## Module Reference

### `icrl/core/types.py` — Data Structures

Everything flows through three dataclasses. Every other module either produces or consumes these.

```python
Transition          # single env step: (obs, action, reward, next_obs, done, cost, info)
Trajectory          # full episode: list[Transition] + total_reward + total_cost + metadata
DemoDataset         # safe: list[Trajectory], unsafe: list[Trajectory]
                    # .safe_reward_threshold → the adversarial pivot
```

`DemoDataset.safe_reward_threshold` is a computed property — the single most important number in the whole system:
```python
@property
def safe_reward_threshold(self) -> float:
    return max(t.total_reward for t in self.safe)
```

### `icrl/core/interfaces.py` — Abstract Base Classes

Four ABCs define every cross-module contract. Implementing any one of them and passing it to `ICRLTrainer` replaces that component without touching anything else.

| ABC | Key methods | What it represents |
|---|---|---|
| `BaseEnv` | `reset`, `step`, `obs_repr`, `action_repr` | The world the agent acts in |
| `BaseEmbedder` | `embed_trajectory(traj, env)` → `Tensor[D]` | Trajectory → fixed vector |
| `BaseConstraint` | `feasibility(traj)`, `cost(traj)`, `update(safe, unsafe)` | Learned safety function |
| `BasePolicy` | `act(obs)`, `update(trajectories)`, `set_constraint(c)` | The agent |

`obs_repr` and `action_repr` on `BaseEnv` are the bridge between numeric RL state and the text embedding — this is how the constraint generalises from GridWorld coordinates to DOM trees and API call logs.

### `icrl/trainer/adversarial_detector.py` — The Adversarial Principle

```python
class AdversarialDetector:
    def fit(self, demos: DemoDataset) -> float:
        # computes threshold from safe demo rewards
        # mode: "max" | "mean_plus_std" | "percentile"

    def is_unsafe(self, trajectory: Trajectory) -> bool:
        return trajectory.total_reward > self.threshold + self.config.min_reward_gap

    def filter_unsafe(self, trajectories) -> (safe, unsafe):
        # splits a batch into (safe_enough, flagged_as_unsafe)
```

Three threshold modes address different assumptions about how near-optimal the demos are:
- **`max`** — strict: any trajectory beating the best demo is unsafe (works when demos are truly optimal)
- **`mean_plus_std`** — lenient: allows for some demo variance; useful when demos have inconsistent quality
- **`percentile`** — configurable: e.g., p=95 means only the top 5% of safe demos set the bar

### `icrl/embedders/mean_pool.py` — Trajectory Embedding

Converts a trajectory to a vector the constraint MLP can reason about.

**Pipeline:**
```
for each transition in trajectory:
    text = "OBS pos=(2,0) confirmed=False data=False ACT access_data"
    tokens = text.split()
    lookup each token in learned embedding table
    → Tensor[embed_dim] per token

mean-pool over all tokens across all transitions
→ Tensor[embed_dim]
```

The embedding table is trained jointly with the constraint MLP — there is no frozen pre-trained model in the toy setup. This is sufficient for GridWorld and SimWeb because the vocabulary is small and structured.

For real web tasks where actions are LLM-generated text and observations are DOM trees, swap to `SentenceTransformerEmbedder` (implements the same `BaseEmbedder` interface).

**Why text and not raw obs/action tensors?**
Concatenating `[x, y, confirmed, data_accessed, action_id]` would hard-code the GridWorld state structure into the constraint. Using `obs_repr` and `action_repr` as text means the same constraint architecture handles:
- GridWorld: `"pos=(2,0) confirmed=False"` 
- Web agent: `"url=https://crm/contacts page_title=Edit confirmed=False"`
- Tool call: `"tool=delete_record args={'id': 42} response=pending"`

### `icrl/constraints/mlp_constraint.py` — Constraint Function

```python
C_θ(τ) = sigmoid(MLP(concat(embed(τ), reward_norm, length_norm)))
```

The reward and length features are concatenated because the adversarial principle directly links reward level to unsafety. The MLP can learn: "high reward + trajectory that never called confirm = unsafe."

**Training (contrastive loss):**
```
L = −mean(log C(τ_safe)) − mean(log(1 − C(τ_unsafe)))
```
This is the offline version of the standard ICRL gradient update. Because the adversarial detector converts learner trajectories into an explicit `D_unsafe` set, the expectation over the learner policy becomes an offline sample — no live policy gradient needed during the constraint update step.

**Three loss options** (set in config):
- `contrastive` — binary cross-entropy, default, works well in most settings
- `margin` — enforces `C(safe) − C(unsafe) ≥ margin`; better when reward ranges overlap
- `info_nce` — representation-level contrastive; useful when scaling to larger models

### `icrl/policies/ppo_lagrangian.py` — Constrained Policy

Standard PPO with one addition: a Lagrange multiplier λ that converts the constraint into a soft penalty on the policy gradient.

```
combined_advantage = adv_reward − λ · adv_cost
λ  ←  clip(λ + lr_lag · (mean_episode_cost − cost_limit), 0, λ_max)
```

Episode cost is `−log C_θ(τ)` (from the constraint), distributed uniformly across timesteps for the GAE computation. As the constraint learns to score unsafe trajectories poorly, their cost rises, λ rises, and the policy is pushed toward safer behaviour.

The network has two critic heads: one for reward value, one for cost value. This allows separate GAE estimation for reward and cost advantages — standard practice in constrained RL (following the CPO/PPO-Lag formulation).

### `icrl/trainer/icrl_trainer.py` — Main Loop

Owns the training loop and wires all components together. Accepts:

```python
ICRLTrainer(
    env        : BaseEnv,
    policy     : BasePolicy,
    constraint : BaseConstraint,
    demos      : DemoDataset,
    detector   : AdversarialDetector,
    config     : ICRLConfig,
)
```

Key config parameters:

| Parameter | What it does |
|---|---|
| `pretrain_iterations` | Run unconstrained PPO first; constraint activates after |
| `min_unsafe_for_update` | Don't train constraint until this many unsafe examples accumulate |
| `constraint_batch_size` | Half from safe demos, half from unsafe buffer each constraint step |
| `unsafe_buffer_max_size` | FIFO cap on the unsafe buffer; older examples are dropped |
| `n_constraint_epochs` | Constraint gradient steps per iteration |

---

## Environments

### `GridWorldEnv` (toy, implemented)

An N×N grid designed to make the adversarial principle concrete and testable.

```
Start (0,0) ──► Data (N-1, 0) ──► Goal (N-1, N-1)
```

| Path | Steps | Reward (N=3, C=20) |
|---|---|---|
| Safe: navigate → confirm → access → navigate → submit | 2(N−1)+3 = 7 | 13.0 |
| Unsafe: navigate → access (no confirm) → navigate → submit | 2(N−1)+2 = 6 | **14.0** |

The unsafe path always scores exactly 1 higher than the safe path. The adversarial detector threshold is 13.0 (max safe demo reward). Any completed trajectory with reward ≥ 14.01 is automatically flagged.

**Text actions** mirror web-agent primitives:
`move_north/south/east/west`, `request_confirmation`, `access_data`, `submit`

**Numeric state** for the policy MLP:
`[x, y, confirmed (0/1), data_accessed (0/1)]`

### `SimWebEnv` (next milestone, not yet implemented)

A state-machine simulation of web tasks with text observations and actions. Steps between GridWorld (structured numeric state) and real ST-WebAgentBench (browser + DOM). Add it by implementing `BaseEnv`.

### ST-WebAgentBench (target, not yet implemented)

Real enterprise web tasks in a browser. Wrap BrowserGym's ST-WebAgentBench environment in a `BaseEnv` adapter. The only change needed is `obs_repr` (DOM → text) and `action_repr` (browser action → text), and switching the embedder to `SentenceTransformerEmbedder`.

---

## Running Experiments

**Install:**
```bash
pip install -e ".[dev]"
```

**Run the smoke test:**
```bash
python3.11 scripts/gridworld/run_gridworld_icrl_loop.py
```

**Run baseline (no constraint):**
```bash
python3.11 scripts/gridworld/run_gridworld_icrl_loop.py --no-constraint
```

**Watch metrics in real time:**
```bash
tail -f runs/gridworld_smoke/metrics.jsonl | python3.11 -c "
import sys, json
for line in sys.stdin:
    m = json.loads(line)
    print(f\"iter={m['iteration']:4d}  reward={m.get('mean_reward',0):7.2f}  "
          f\"unsafe_buf={m.get('unsafe_buffer_size',0):3d}  "
          f\"c_loss={m.get('constraint_loss','-')!s:8}  "
          f\"gap={m.get('feasibility_gap','-')!s:6}  "
          f\"lam={m.get('lambda','-')!s:5}\")
"
```

**Run tests:**
```bash
python3.11 -m pytest tests/ -v
```

---

## How to Extend

### Add a new environment

1. Implement `BaseEnv` in `icrl/envs/your_env.py`
2. Implement `BaseDemoLoader` in `icrl/demos/your_env_demos.py`
3. Add a config YAML in `configs/experiment/`
4. Write a run script in `scripts/gridworld/run_your_env.py` following `run_gridworld_icrl_loop.py`

The rest of the system (constraint, policy, trainer, detector) is unchanged.

### Add a new constraint architecture

Implement `BaseConstraint` in `icrl/constraints/your_constraint.py`. Minimum interface:
```python
def feasibility(self, traj: Trajectory) -> torch.Tensor: ...  # scalar in [0,1]
def cost(self, traj: Trajectory) -> torch.Tensor: ...         # -log(feasibility)
def update(self, safe, unsafe) -> dict: ...                   # training step
```

Plug it in where `MLPConstraint` is constructed in the run script.

### Add a new embedder (e.g., SentenceTransformer)

Implement `BaseEmbedder` in `icrl/embedders/sentence_transformer.py`:
```python
class SentenceTransformerEmbedder(BaseEmbedder):
    def embed_trajectory(self, traj, env) -> torch.Tensor:
        texts = [f"OBS {env.obs_repr(t.obs)} ACT {env.action_repr(t.action)}"
                 for t in traj.transitions]
        return self.model.encode(texts, convert_to_tensor=True).mean(dim=0)
```

Pass it to `MLPConstraint` in place of `MeanPoolEmbedder`. The constraint MLP input dimension adjusts automatically via `embedder.embed_dim`.

### Swap the constraint loss

Set `loss: margin` or `loss: info_nce` in the config YAML. The three options are in `icrl/constraints/losses.py`. Add a new one there and handle it in `MLPConstraint.update()`.

### Change the threshold mode

Set `detector.mode: mean_plus_std` or `detector.mode: percentile` in the config. Use `mean_plus_std` if your safe demos have variable quality; use `percentile` to be more lenient with near-optimal but not perfectly-optimal demos.

---

## Key Numbers for GridWorld (N=3)

```
Safe demo reward (threshold)  : 13.0  =  20 - 7 steps
Unsafe shortcut reward        : 14.0  =  20 - 6 steps
Adversarial gap               :  1.0

If mean_reward after ICRL → 13.0 : policy learned the safe behaviour ✓
If mean_reward stays at   → 14.0 : constraint not yet learned or too weak ✗

Expected feasibility after training:
  safe trajectory   → C_θ ≈ 0.9+  (high feasibility)
  unsafe trajectory → C_θ ≈ 0.1−  (low feasibility)
  feasibility_gap = safe − unsafe  → should increase monotonically
```

---

## File Quick-Reference

| File | One line |
|---|---|
| `icrl/core/types.py` | `Transition`, `Trajectory`, `DemoDataset` — all shared data |
| `icrl/core/interfaces.py` | ABCs: `BaseEnv`, `BaseEmbedder`, `BaseConstraint`, `BasePolicy` |
| `icrl/trainer/adversarial_detector.py` | Threshold + flagging — the adversarial principle in code |
| `icrl/trainer/icrl_trainer.py` | 4-step loop: collect → detect → constrain → update |
| `icrl/trainer/rollout_buffer.py` | `collect_rollouts(env, policy, n_steps) → list[Trajectory]` |
| `icrl/envs/gridworld.py` | N×N toy env with deliberate reward gap |
| `icrl/demos/gridworld_demos.py` | Generates safe demos via hard-coded safe policy |
| `icrl/embedders/mean_pool.py` | Tokenise trajectory text → learned embedding → mean-pool |
| `icrl/constraints/losses.py` | Contrastive, margin, InfoNCE loss functions |
| `icrl/constraints/mlp_constraint.py` | MLP on trajectory embedding → feasibility score |
| `icrl/policies/ppo_lagrangian.py` | PPO + Lagrange multiplier for constrained optimisation |
| `icrl/policies/rule_based.py` | `SafeRuleBasedPolicy` and `UnsafeRuleBasedPolicy` oracles |
| `icrl/metrics/tracker.py` | Writes `metrics.jsonl` + optional TensorBoard |
| `icrl/utils/config.py` | `load_config(yaml_path) → dict` |
| `configs/experiment/gridworld_smoke.yaml` | Default experiment config |
| `scripts/gridworld/run_gridworld_icrl_loop.py` | Entry point — constructs everything and calls `trainer.train()` |
