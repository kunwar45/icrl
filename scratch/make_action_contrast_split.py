# ABOUTME: Derives a split whose SAFE class is benign completions only, no refusals
# ABOUTME: Run: python scratch/make_action_contrast_split.py <src_split> <dst_split>
"""
Redefine the contrast so trajectory length carries no signal.

Measured 2026-08-19 on the held-out split: a step counter scores AUROC 0.806
against the as-generated expert class, and C_theta scored 0.824 — a gain of
0.018, i.e. nothing. The shortcut is entirely the refusals, which are one action
by construction (median 1 step against unsafe's 4). Rebalancing barely helps
(0.777) because a decline is INTRINSICALLY short.

Dropping refusals from the safe class takes the step-count baseline to 0.554 —
chance. What remains is benign work completed versus hazard completed: both
multi-step, both similar length, differing in WHAT the agent manipulates. That is
the constraint worth learning, and it is the one that survives fine-tuning: a
C_theta that encodes "long = costly" drives Lagrangian PPO toward a policy that
does nothing, while one that encodes "these manipulations are costly" steers away
from the hazard specifically and lets refusal emerge as a consequence.

The refusal traces are not deleted — they are the eval set for whether the tuned
policy actually declines.
"""
import json, sys
from pathlib import Path

src, dst = Path(sys.argv[1]), Path(sys.argv[2])
for sub, name in (("train", "safe.jsonl"), ("eval", "safe_held_out.jsonl")):
    rows = [json.loads(l) for l in (src / sub / name).open()]
    keep = [r for r in rows
            if not str(r.get("task_instance_id") or r.get("task_id") or "").startswith("haz")]
    (dst / sub).mkdir(parents=True, exist_ok=True)
    with (dst / sub / name).open("w") as f:
        for r in keep:
            f.write(json.dumps(r) + "\n")
    print(f"{sub}/{name}: {len(rows)} -> {len(keep)} (dropped {len(rows)-len(keep)} refusals)")

for sub, name in (("train", "unsafe.jsonl"), ("eval", "unsafe_held_out.jsonl")):
    rows = [json.loads(l) for l in (src / sub / name).open()]
    (dst / sub).mkdir(parents=True, exist_ok=True)
    with (dst / sub / name).open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"{sub}/{name}: {len(rows)} (unchanged)")
