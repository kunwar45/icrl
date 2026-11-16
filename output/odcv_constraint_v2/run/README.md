---
license: other
tags:
- inverse-constraint-rl
- constraint-head
- odcv
base_model: Qwen/Qwen2.5-1.5B
---

# 2026-09-04 odcv constraint head — bce, actions-only (numina-control audited split)

The trained constraint head C_θ from the ICRL project: a 2-layer MLP (1536 → 256 → 1, sigmoid) on
mean-pooled **frozen Qwen/Qwen2.5-1.5B** hidden states of the trajectory text serialised in
`text_mode=actions_only` at max_length 4096. Trained with the `bce` objective for 1000 steps
(configs/icrl_dual_training/constraint_bce_actions_only.yaml, commit a6f53ce) on the train half of
`icrl-finetuning/2026-09-04-odcv-numina-control-audited-contrast-split` (186 expert / 216 unsafe, 30 scenarios).

Held-out (10 unseen scenarios, 46 expert / 78 unsafe): AUROC 0.910, F1 0.899, ECE 0.074,
expert mean score 0.23, unsafe mean 0.87. Within-scenario pair-weighted AUROC 0.812.
Trivial baselines on the same traces: protected-path-write regex 0.838 (within 0.755), step count 0.688 (within 0.515).
The v1 head (icrl loss, full text) scored 0.797 / 0.755 with a saturated expert median of 0.997.

Files: `constraint_head.pt` (state_dict of the head; load with `load_constraint_head` from
`src/trajectory_embedding/trajectory_encoder.py`, which restores `text_mode` from `constraint_head.meta.json`),
`train_metrics.json`, `held_out_metrics.json` (per-trace scores and scenario ids).
