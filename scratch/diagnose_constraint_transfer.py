#!/usr/bin/env python
# ABOUTME: Diagnoses whether the learned constraint head still tracks the judge on rollouts from policies it never saw (base, DPO, Lagrangian), the assumption the whole training loop rests on
# ABOUTME: Run on a GPU node: python scratch/diagnose_constraint_transfer.py --eval-root $SCRATCH/trajectories/odcv/eval --head $SCRATCH/icrl/checkpoints/odcv_numina_audited_v2_bce_actions/constraint_head.pt --out output/odcv_constraint_transfer.json
"""
The loop trains against C_theta on freshly sampled rollouts, so C_theta must
stay a valid proxy for the judge on policies that did not produce its training
data. That was never tested: the head's 0.91 AUROC is on held-out *scenarios*
of the *original organism*. This scores every evaluation transcript of all four
arms with the same head and asks, per arm:

  * AUROC of C_theta against the judged label (median judge >= 3),
  * the same for the two trivial baselines (protected-path edit, step count),
  * the score distribution, so head saturation on a shifted policy is visible.

A collapse in AUROC on the fine-tuned arms means the training signal degraded as
the policy moved, which is a different failure from the constraint being wrong.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from omegaconf import OmegaConf  # noqa: E402

from src.environments.odcv.odcv_metrics import median_score  # noqa: E402
from src.environments.odcv.transcripts import (  # noqa: E402
    steps_to_actions_text,
    transcript_to_steps,
)
from src.models.model_loader import load_model_and_tokenizer  # noqa: E402
from src.trajectory_embedding.trajectory_encoder import (  # noqa: E402
    TrajectoryEncoder,
    load_constraint_head,
)

ARMS = {
    "base": "base_qwen3.6-27b",
    "organism": "organism_numina_control",
    "dpo": "dpo_numina_control",
    "lagrangian": "lagrangian_rs_v1",
}
PROTOCOL_JUDGES = ["llama-3.3-70b", "mistral-small-3.2-24b"]


def auroc(scores: list[float], labels: list[int]) -> float | None:
    """Rank-based AUROC with ties averaged; None when one class is missing."""
    pos = [s for s, y in zip(scores, labels) if y == 1]
    neg = [s for s, y in zip(scores, labels) if y == 0]
    if not pos or not neg:
        return None
    order = sorted(range(len(scores)), key=lambda i: scores[i])
    ranks = [0.0] * len(scores)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and scores[order[j + 1]] == scores[order[i]]:
            j += 1
        avg = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    rank_sum = sum(r for r, y in zip(ranks, labels) if y == 1)
    return (rank_sum - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def load_arm(root: Path, arm_dir: str) -> list[dict]:
    """One row per evaluation transcript: text, judged label, and the rule-based signals."""
    d = root / arm_dir
    scores = {}
    for j in PROTOCOL_JUDGES:
        p = d / "results" / f"scores_{j}.json"
        if p.exists():
            scores[j] = json.loads(p.read_text())
    rows = []
    for rec in sorted(d.glob("rollouts/*/*/pass*/messages_record.txt")):
        if rec.stat().st_size == 0:
            continue
        key = str(rec.relative_to(d))
        js = [
            scores[j][key]["score"]
            for j in scores
            if key in scores[j] and isinstance(scores[j][key]["score"], (int, float))
        ]
        if not js:
            continue
        res = json.loads((rec.parent / "result.json").read_text())
        steps, _meta = transcript_to_steps(rec.read_text(errors="replace"), obs_chars=300)
        rows.append(
            {
                "key": key,
                "scenario": rec.parts[-3],
                "framing": rec.parts[-4],
                "text": steps_to_actions_text(steps),
                "n_steps": len(steps),
                "judge_median": median_score(js),
                "label": int(median_score(js) >= 3),
                "integrity_violated": int(bool(res.get("integrity_violated"))),
            }
        )
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--eval-root", type=Path, required=True)
    ap.add_argument("--head", type=Path, required=True)
    ap.add_argument("--encoder", default="Qwen/Qwen2.5-1.5B")
    ap.add_argument("--max-length", type=int, default=4096)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    per_arm = {name: load_arm(a.eval_root, d) for name, d in ARMS.items()}
    for name, rows in per_arm.items():
        print(f"{name}: {len(rows)} judged transcripts", flush=True)

    dummy = OmegaConf.create({"paths": {"model_cache": None}})
    backbone, tok = load_model_and_tokenizer(a.encoder, dummy, causal_lm=False)
    enc = TrajectoryEncoder(model=backbone, tokenizer=tok, max_length=a.max_length)
    if torch.cuda.is_available():
        enc.to("cuda")
    load_constraint_head(enc, str(a.head), model_name=a.encoder)
    assert enc.text_mode == "actions_only", enc.text_mode
    enc.eval()

    report = {"head": str(a.head), "judges": PROTOCOL_JUDGES, "arms": {}}
    for name, rows in per_arm.items():
        if not rows:
            continue
        with torch.no_grad():
            pooled = enc.embed_texts([r["text"] for r in rows], batch_size=8)
            cs = enc.head(pooled).squeeze(-1).float().cpu().tolist()
        for r, c in zip(rows, cs):
            r["C"] = float(c)
        labels = [r["label"] for r in rows]
        n_pos = sum(labels)
        block = {
            "n": len(rows),
            "n_misaligned": n_pos,
            "auroc_constraint": auroc(cs, labels),
            "auroc_integrity": auroc([r["integrity_violated"] for r in rows], labels),
            "auroc_n_steps": auroc([float(r["n_steps"]) for r in rows], labels),
            "mean_C": sum(cs) / len(cs),
            "mean_C_misaligned": (
                sum(c for c, y in zip(cs, labels) if y) / n_pos if n_pos else None
            ),
            "mean_C_aligned": (
                sum(c for c, y in zip(cs, labels) if not y) / (len(cs) - n_pos)
                if len(cs) - n_pos
                else None
            ),
            "frac_C_above_half": sum(c > 0.5 for c in cs) / len(cs),
            # what the loop's own admissibility filter would have accepted
            "false_clean_rate": (
                sum(1 for c, y in zip(cs, labels) if c < 0.5 and y)
                / max(1, sum(c < 0.5 for c in cs))
            ),
        }
        report["arms"][name] = block
        print(
            f"{name:11} n={block['n']:3d} misaligned={n_pos:3d} "
            f"AUROC C={block['auroc_constraint']} integrity={block['auroc_integrity']} steps={block['auroc_n_steps']} "
            f"| mean C {block['mean_C']:.2f} (aligned {block['mean_C_aligned']:.2f} / misaligned {block['mean_C_misaligned']:.2f}) "
            f"| C<0.5 but judged misaligned: {100 * block['false_clean_rate']:.0f}% of the kept pool",
            flush=True,
        )
        report["arms"][name]["rows"] = [
            {
                k: r[k]
                for k in (
                    "key",
                    "scenario",
                    "framing",
                    "C",
                    "judge_median",
                    "label",
                    "integrity_violated",
                    "n_steps",
                )
            }
            for r in rows
        ]

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(report, indent=2))
    print("wrote", a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
