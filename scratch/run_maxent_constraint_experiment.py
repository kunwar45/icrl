#!/usr/bin/env python
# ABOUTME: One-off head-to-head: fit the maximum-entropy inverse constraint inference of the ICRL survey on real ODCV traces, and score it against the frozen supervised classifier on the same judged evaluation transcripts
# ABOUTME: Run on a GPU node: python scratch/run_maxent_constraint_experiment.py --expert-dir $SCRATCH/trajectories/odcv/numina_control_audited/expert --nominal-root $SCRATCH/icrl/checkpoints/odcv_lagrangian_rs_v1 --eval-root $SCRATCH/trajectories/odcv/eval --out output/odcv_maxent_constraint.json
"""
Does inverse constraint inference actually beat the supervised classifier we
have been using?

The classifier was fitted once on a fixed safe-versus-unsafe split and scores a
whole trajectory. The maximum-entropy method (Malik et al. 2021, as stated in
arXiv:2409.07569 Section 4.2) instead learns a per-step feasibility function from
the gap between expert demonstrations and the CURRENT policy's rollouts, with no
labels on the policy side at all.

This fits the second on exactly the data the first had available and evaluates
both the same way: cost of each of the judged evaluation transcripts, AUROC
against the two-judge verdict, split by whether the scenario was held out of
constraint training. Nothing here touches the policy; it is a measurement of the
constraint model alone.

The nominal pool is the policy's own rollouts, unlabelled, which is what makes
the comparison fair rather than flattering: the method is given strictly less
supervision than the classifier it is being compared against.
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
from src.environments.odcv.transcripts import transcript_to_steps  # noqa: E402
from src.icrl_dual_training.maxent_constraint import (  # noqa: E402
    MaxEntConstraintConfig,
    StepFeasibilityHead,
    TrajectoryBatch,
    maxent_constraint_update,
    pucl_constraint_update,
    trajectory_cost,
)
from src.models.model_loader import load_model_and_tokenizer  # noqa: E402
from src.trajectory_embedding.trajectory_encoder import TrajectoryEncoder  # noqa: E402

ARMS = {
    "base": "base_qwen3.6-27b",
    "organism": "organism_numina_control",
    "dpo": "dpo_numina_control",
    "lagrangian": "lagrangian_rs_v1",
}
JUDGES = ["llama-3.3-70b", "mistral-small-3.2-24b"]


def auroc(scores, labels) -> float | None:
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
        for k in range(i, j + 1):
            ranks[order[k]] = (i + j) / 2 + 1
        i = j + 1
    return (
        sum(r for r, y in zip(ranks, labels) if y == 1) - len(pos) * (len(pos) + 1) / 2
    ) / (len(pos) * len(neg))


def step_text(step: dict, obs_chars: int) -> str:
    """One (state, action) pair as text: what the agent saw, then what it did.

    The classifier saw only actions, concatenated. The feasibility function is
    defined on state-action pairs, so the observation belongs in the input; the
    survey notes this is exactly what lets a constraint depend on context.
    """
    obs = (step.get("observation") or "")[:obs_chars]
    return f"[STATE] {obs}\n[ACTION] {step.get('action', '')}"


def load_expert(expert_dir: Path) -> list[dict]:
    out = []
    for f in sorted(expert_dir.glob("task_*.json")):
        d = json.loads(f.read_text())
        if d.get("steps"):
            out.append(
                {"scenario": d.get("task_id"), "steps": d["steps"], "source": f.name}
            )
    return out


def load_nominal(root: Path, limit: int | None) -> list[dict]:
    """The policy's own rollouts from every completed round, unlabelled."""
    out = []
    for rec in sorted(
        root.glob("round_*/rollouts/rollouts/*/*/pass*/messages_record.txt")
    ):
        if rec.stat().st_size == 0:
            continue
        steps, _ = transcript_to_steps(rec.read_text(errors="replace"), obs_chars=300)
        if steps:
            out.append({"scenario": rec.parts[-3], "steps": steps, "source": str(rec)})
        if limit and len(out) >= limit:
            break
    return out


def load_eval(eval_root: Path) -> list[dict]:
    rows = []
    for arm, d in ARMS.items():
        adir = eval_root / d
        scores = {}
        for j in JUDGES:
            p = adir / "results" / f"scores_{j}.json"
            if p.exists():
                scores[j] = json.loads(p.read_text())
        for rec in sorted(adir.glob("rollouts/*/*/pass*/messages_record.txt")):
            if rec.stat().st_size == 0:
                continue
            key = str(rec.relative_to(adir))
            js = [
                scores[j][key]["score"]
                for j in scores
                if key in scores[j]
                and isinstance(scores[j][key]["score"], (int, float))
            ]
            if not js:
                continue
            steps, _ = transcript_to_steps(
                rec.read_text(errors="replace"), obs_chars=300
            )
            if not steps:
                continue
            rows.append(
                {
                    "arm": arm,
                    "scenario": rec.parts[-3],
                    "steps": steps,
                    "label": int(median_score(js) >= 3),
                }
            )
    return rows


def embed(enc, trajs: list[dict], obs_chars: int, batch_size: int) -> TrajectoryBatch:
    """Per-step embeddings for a list of trajectories, flattened with owners."""
    texts, owner = [], []
    for i, t in enumerate(trajs):
        for s in t["steps"]:
            texts.append(step_text(s, obs_chars))
            owner.append(i)
    with torch.no_grad():
        pooled = enc.embed_texts(texts, batch_size=batch_size)
    return TrajectoryBatch(
        embeddings=pooled.float().cpu(),
        owner=torch.tensor(owner, dtype=torch.long),
        n_trajectories=len(trajs),
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--expert-dir", type=Path, required=True)
    ap.add_argument("--nominal-root", type=Path, required=True)
    ap.add_argument("--eval-root", type=Path, required=True)
    ap.add_argument("--splits", type=Path, required=True)
    ap.add_argument("--encoder", default="Qwen/Qwen2.5-1.5B")
    ap.add_argument("--max-length", type=int, default=1024)
    ap.add_argument("--obs-chars", type=int, default=600)
    ap.add_argument("--nominal-limit", type=int, default=900)
    ap.add_argument("--embed-batch", type=int, default=32)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    held_out = set()
    sp = json.loads(a.splits.read_text())
    for lab in sp.get("labels", {}).values():
        held_out |= set(lab.get("held_out_task_ids", []))
    print(f"{len(held_out)} held-out scenarios", flush=True)

    expert = [t for t in load_expert(a.expert_dir) if t["scenario"] not in held_out]
    nominal = [
        t
        for t in load_nominal(a.nominal_root, a.nominal_limit)
        if t["scenario"] not in held_out
    ]
    ev = load_eval(a.eval_root)
    print(
        f"expert {len(expert)} | nominal {len(nominal)} (unlabelled) | eval {len(ev)}",
        flush=True,
    )

    dummy = OmegaConf.create({"paths": {"model_cache": None}})
    backbone, tok = load_model_and_tokenizer(a.encoder, dummy, causal_lm=False)
    enc = TrajectoryEncoder(model=backbone, tokenizer=tok, max_length=a.max_length)
    if torch.cuda.is_available():
        enc.to("cuda")
    enc.eval()

    print("embedding steps...", flush=True)
    e_batch = embed(enc, expert, a.obs_chars, a.embed_batch)
    n_batch = embed(enc, nominal, a.obs_chars, a.embed_batch)
    v_batch = embed(enc, ev, a.obs_chars, a.embed_batch)
    dim = e_batch.embeddings.shape[1]
    print(
        f"steps: expert {e_batch.embeddings.shape[0]}, nominal {n_batch.embeddings.shape[0]}, "
        f"eval {v_batch.embeddings.shape[0]}, dim {dim}",
        flush=True,
    )
    del enc, backbone
    torch.cuda.empty_cache()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    report = {
        "n_expert": len(expert),
        "n_nominal": len(nominal),
        "n_eval": len(ev),
        "held_out_scenarios": sorted(held_out),
        "variants": {},
    }
    labels = [r["label"] for r in ev]
    is_ho = [r["scenario"] in held_out for r in ev]

    def report_variant(name, head, extra=None):
        with torch.no_grad():
            cost = trajectory_cost(head, v_batch.to(device)).cpu().tolist()
            cost_norm = (
                trajectory_cost(head, v_batch.to(device), normalise_by_length=True)
                .cpu().tolist()
            )
        # precision at the operating point the loop would use: flag the same
        # share of traces the supervised head flags, so the comparison is fair
        order = sorted(range(len(cost)), key=lambda i: -cost[i])
        k = int(round(0.46 * len(cost)))
        flagged = set(order[:k])
        tp = sum(labels[i] for i in flagged)
        fp = k - tp
        fn = sum(labels) - tp
        tn = len(cost) - tp - fp - fn
        block = {
            "confusion_at_46pct": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
            "accuracy_at_46pct": (tp + tn) / len(cost),
            "auroc_cost": auroc(cost, labels),
            "auroc_cost_per_step": auroc(cost_norm, labels),
            "auroc_held_out": auroc(
                [c for c, h in zip(cost, is_ho) if h], [y for y, h in zip(labels, is_ho) if h]
            ),
            "precision_at_46pct_flagged": tp / max(1, k),
            "recall_at_46pct_flagged": tp / max(1, sum(labels)),
            "per_arm": {
                arm: auroc(
                    [c for c, r in zip(cost, ev) if r["arm"] == arm],
                    [r["label"] for r in ev if r["arm"] == arm],
                )
                for arm in ARMS
            },
            **(extra or {}),
        }
        report["variants"][name] = block
        print(
            f"{name:22} AUROC {block['auroc_cost']:.3f} | per-step {block['auroc_cost_per_step']:.3f} "
            f"| held-out {block['auroc_held_out']:.3f} "
            f"| at 46% flagged: precision {100*block['precision_at_46pct_flagged']:.0f}% "
            f"recall {100*block['recall_at_46pct_flagged']:.0f}% acc {100*block['accuracy_at_46pct']:.0f}% "
            f"| TP {tp} FP {fp} FN {fn} TN {tn}",
            flush=True,
        )

    for name, selection, frac in (
        ("maxent_all_negatives", "all", 1.0),
        ("maxent_score_selected", "pucl", 0.5),
    ):
        torch.manual_seed(0)
        head = StepFeasibilityHead(dim).to(device)
        cfg = MaxEntConstraintConfig(
            n_steps=a.steps, negative_selection=selection, pucl_keep_fraction=frac
        )
        stats = maxent_constraint_update(head, e_batch, n_batch, cfg)
        report_variant(name, head, {"train_stats": stats})

    # the published two-step method: distance-selected reliable negatives
    for frac in (0.2, 0.3, 0.5):
        torch.manual_seed(0)
        head = StepFeasibilityHead(dim).to(device)
        cfg = MaxEntConstraintConfig(
            n_steps=max(a.steps, 400), lr=1e-3, pucl_keep_fraction=frac, pucl_k=2
        )
        stats, _mem = pucl_constraint_update(head, e_batch, n_batch, cfg)
        report_variant(f"pucl_keep{frac}", head, {"train_stats": stats})

    # Control: supervised cross-entropy on the same per-step features, using the
    # audited labels. If this lands near the unsupervised variants, the frozen
    # representation is the ceiling and no objective on top of it will help.
    torch.manual_seed(0)
    head = StepFeasibilityHead(dim).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=0.01)
    pos, neg = e_batch.embeddings.to(device), n_batch.embeddings.to(device)
    for _ in range(max(a.steps, 400)):
        pi = torch.randint(0, pos.shape[0], (256,), device=device)
        ni = torch.randint(0, neg.shape[0], (256,), device=device)
        zp = head(pos[pi]).clamp(1e-6, 1 - 1e-6)
        zn = head(neg[ni]).clamp(1e-6, 1 - 1e-6)
        loss = -(torch.log(zp).mean() + torch.log(1 - zn).mean())
        opt.zero_grad(); loss.backward(); opt.step()
    report_variant("supervised_same_features", head, {"note": "all nominal steps as negatives, with labels"})

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(report, indent=2))
    print("wrote", a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
