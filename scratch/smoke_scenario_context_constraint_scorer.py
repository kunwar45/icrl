#!/usr/bin/env python
# ABOUTME: Loads the saved ODCV constraint model through the src loader the fine-tuning loop uses and scores a handful of real transcripts, so a 4-round chain never dies on a scoring bug
# ABOUTME: Run on a GPU node: python scratch/smoke_scenario_context_constraint_scorer.py --model-dir $SCRATCH/constraint_models/odcv_context_rollouts --eval-root $SCRATCH/trajectories/odcv/eval
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.environments.odcv.transcripts import transcript_to_steps  # noqa: E402
from src.icrl_dual_training.scenario_context_constraint import (  # noqa: E402
    load_constraint_model,
    score_items,
)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-dir", type=Path, required=True)
    ap.add_argument("--eval-root", type=Path, required=True)
    ap.add_argument("--n", type=int, default=16)
    a = ap.parse_args()

    model, tok, cfg = load_constraint_model(a.model_dir)
    print(f"loaded constraint model: {cfg['encoder']} arm={cfg['arm']} "
          f"threshold={cfg['threshold']:.4f}", flush=True)

    adir = next(p for p in sorted(a.eval_root.iterdir()) if (p / "rollouts").exists())
    scores_path = adir / "results" / "scores_llama-3.3-70b.json"
    judge = json.loads(scores_path.read_text()) if scores_path.exists() else {}

    items, truth = [], []
    for rec in sorted(adir.glob("rollouts/*/*/pass*/messages_record.txt"))[: a.n]:
        if rec.stat().st_size == 0:
            continue
        steps, meta = transcript_to_steps(rec.read_text(errors="replace"), obs_chars=300)
        if not steps:
            continue
        items.append({"steps": steps, "scenario_prompt": meta["scenario_prompt"]})
        j = judge.get(str(rec.relative_to(adir)), {}).get("score")
        truth.append(j if isinstance(j, (int, float)) else None)

    s = score_items(model, tok, cfg, items, batch=2)
    assert len(s) == len(items), "one score per trajectory"
    assert all(0.0 <= x <= 1.0 for x in s), f"scores must be probabilities: {s[:5]}"
    for x, j in zip(s, truth):
        flag = "VIOLATION" if x >= cfg["threshold"] else "safe     "
        print(f"  C={x:.3f}  {flag}   judge={j}")
    print(f"\nOK: scored {len(s)} transcripts, "
          f"{sum(1 for x in s if x >= cfg['threshold'])} flagged", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
