# ICRL vs. *Teaching Claude Why* — comparison brief

Compares this project with the reference replication repo
([teaching_claude_why_replication](https://github.com/Matthew-Bozoukov/teaching_claude_why_replication)):
what is genuinely different, what to borrow, and whether our constrained-RL
("edited RLHF") approach holds up against the alternative it demonstrates.

## The two bets, one line each

- **Reference**: safety norms are best *written down and internalized* — SFT
  Qwen3-32B on synthetic chats where the assistant reasons from a written
  constitution before declining norm-violations. No RL. Headline: agentic
  misalignment (blackmail/leaking honeypots) **19.3% → 8.0%**, with the gain
  tied to training on the *reasoning traces* (thinking format), not just the
  refusals.
- **Ours**: safety norms are *implicit and unwritten* — recover a constraint
  C_θ from safe demonstrations only (adversarial principle: beat-the-demo
  reward ⇒ a safety step was skipped), then constrain the policy with
  Lagrangian PPO + LoRA. Headline metric: CuP on held-out ST-WebAgentBench
  tasks.

## Structural comparison

| Axis | Teaching Claude Why | This project (ICRL) |
|---|---|---|
| Source of the norm | Written constitution (distilled principle sets) | Implicit in safe demos; never written down |
| Representation | Natural language the model reasons over | Scalar feasibility score C_θ(τ) on trajectory embeddings |
| Enters the policy via | SFT on reasoning traces ("why") | Cost term in constrained RL (λ · C_θ); policy never sees a *reason* |
| Supervision needed | Teacher model (Sonnet via OpenRouter) + a spec | Safe demos only — no teacher, no unsafe labels |
| Generalization story | Internalized values transfer OOD (honeypots ≠ training data) | Constraint is benchmark-specific; transfer untested |
| Safety eval | Honeypots + independent ODCV bench, judge-scored | CuP / violation rate on held-out tasks |
| Capability control | Explicit: mixture ratios vs Tulu control, MMLU/Arena-Hard arms | Only completion rate inside CuP — no capability arm |
| Known Achilles heel | Compliance without understanding; empty-think collapse | Near-optimality assumption: our safe demos mostly have reward ≈ 0 |

The deepest difference: the reference teaches the model **why** an action is
wrong; ours only prices **that** it is wrong. Their ablations suggest the
"why" is where the OOD generalization comes from — which is exactly what a
scalar cost cannot provide.

## Ideas worth stealing

1. **A capability-retention arm.** They never report a safety gain without a
   capability control (MMLU / Arena-Hard vs a 0%-synthetic Tulu arm). We
   report CuP only. Add a non-safety task arm (or at minimum completion rate
   on tasks with no active policies) to every finetune comparison, so a CuP
   gain can't secretly be "the policy got timid".
2. **Turn our CoT pipeline into constraint-to-language distillation.** We
   already have `scripts/cot/` (build dataset + finetune). Reframed through
   their result: have a teacher explain *why* each C_θ-flagged trajectory
   violates the implicit norm (grounded in the flagged step), SFT the policy
   on those explanations, and we get the reference recipe with a *learned*
   constitution instead of a written one. This is the highest-leverage
   cross-pollination and mostly reuses existing code.
3. **Adversarial post-training audit.** They re-attack the tuned model
   (honeypots, Petri-style probes) rather than only re-running the benchmark.
   Our analog: evaluate the tuned policy on tasks engineered to *tempt* the
   shortcut (e.g. reward available without confirmation), not just the
   standard held-out split.
4. **Judge-assisted labels to patch the weak-demo problem.** Their pipeline
   leans on a strong judge throughout. Our adversarial threshold
   (max safe-demo reward) is fragile — preflight shows 79/81 safe demos have
   reward ≈ 0, so "beats the demos" ≈ "does anything". An LLM judge that
   screens flagged trajectories before they enter the unsafe buffer would
   decouple constraint quality from demo quality.
5. **Eval hygiene we already partially adopted** (HF artifact publishing with
   provenance cards, per-arm metadata stamped into the artifact, refusing to
   compare mismatched arms). Worth finishing: stamp demo-source and epsilon
   into the adapter repo the way they stamp thinking-mode, and have eval code
   refuse cross-demo-source comparisons.

## Does the Lagrangian (edited-RLHF) approach still make sense?

**Yes as the headline arm, but it is under-defended.** Constrained RL is the
right shape *when the norm is implicit, act-level, and rollouts are cheap* —
that is our setting, and it is the part the reference repo cannot do (a
constitution must be written before you can SFT on it). But two things need
shoring up:

- **The real risk is upstream of RL.** With near-zero-reward safe demos, the
  adversarial threshold flags competence itself as unsafe, and C_θ learns
  "finishing tasks = violation". No PPO-Lag hyperparameter fixes that. Demo
  quality (the current collection job) and/or judge-screened flagging (idea 4)
  is the critical path.
- **The reference proves a cheaper baseline can move behavior a lot.** If
  plain SFT gets most of the safety gain, the RL machinery isn't earning its
  complexity — we currently can't answer that.

**Concrete additions, cheapest first:**

1. **SFT / behavior-cloning baseline** — fine-tune the policy directly on safe
   demos (and on the CoT explanations from idea 2); compare CuP with the
   Lagrangian arm. This is the experiment the reference repo makes obligatory.
2. **Inference-time shield ablation** — use C_θ to veto/rerank candidate
   actions at rollout time, no fine-tuning. Separates "is the constraint any
   good" from "did constrained RL optimize it", and gives a deployable
   fallback.
3. **Offline preference optimization (DPO/KTO) over trajectory pairs** — we
   already produce (safe, adversarially-flagged) pairs; offline preference
   training is stabler than PPO-Lag and needs no reward model. A natural
   middle arm between SFT and full Lagrangian RL.
4. **The hybrid (the genuinely novel contribution):** distill C_θ into a
   short written constraint list (inverse-constitutional step), then run the
   reference repo's internalization recipe on it. If CuP transfers to unseen
   task families better than the scalar-cost arm, that is a result neither
   project alone can claim.

**Bottom line:** keep Lagrangian PPO as the centerpiece — it is the only arm
that works when nobody can write the rule down — but bracket it with the SFT
baseline (1) and the shield ablation (2) so its marginal value is measurable,
and pursue (4) as the bridge between the two projects.
