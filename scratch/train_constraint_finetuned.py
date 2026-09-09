#!/usr/bin/env python
# ABOUTME: Trains the constraint function with the encoder unfrozen (LoRA) instead of frozen, and scores it against the frozen control on the same judged evaluation traces
# ABOUTME: Run on a GPU node: python scratch/train_constraint_finetuned.py --expert-dir <dir> --unsafe-dir <dir> --eval-root <dir> --splits <splits.json> --out output/odcv_constraint_finetuned.json
"""
Every objective we have tried lands at 30-40% precision on the same frozen
encoder: supervised cross-entropy with the full audited label set (0.805 AUROC,
40% precision), maximum-entropy inverse inference (0.685), and positive-unlabelled
constraint learning (0.750, 36%). Crucially, supervised training on the identical
per-step features PUCL uses reaches 35% -- the labels bought nothing. When the
objective stops mattering, the representation is the ceiling.

This tests that directly. Same data, same head, same evaluation; the only change
is that the encoder's attention and MLP projections get LoRA adapters and are
trained with the head rather than held fixed. The `--freeze` control runs the
identical code path with the adapters disabled, so the comparison isolates one
variable.

Reported on the 625 judged evaluation transcripts from four policies: AUROC, and
the confusion matrix at a matched flag rate so precision is comparable across
runs rather than read at whatever threshold flatters each.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.environments.odcv.odcv_metrics import median_score  # noqa: E402
from src.environments.odcv.transcripts import (  # noqa: E402
    steps_to_actions_text,
    transcript_to_steps,
)

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


class ConstraintModel(nn.Module):
    """Encoder plus a pooled head. The encoder is either frozen or LoRA-trained."""

    def __init__(self, encoder, hidden: int, head_hidden: int = 256):
        super().__init__()
        self.encoder = encoder
        self.head = nn.Sequential(
            nn.Linear(hidden, head_hidden), nn.GELU(), nn.Linear(head_hidden, 1)
        )

    def forward(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        h = out.last_hidden_state
        m = attention_mask.unsqueeze(-1).to(h.dtype)
        pooled = (h * m).sum(1) / m.sum(1).clamp(min=1e-6)
        return self.head(pooled.float()).squeeze(-1)  # logit


def load_training(expert_dir: Path, unsafe_dir: Path) -> list[dict]:
    rows = []
    for d, label in ((expert_dir, 0), (unsafe_dir, 1)):
        for f in sorted(d.glob("task_*.json")):
            t = json.loads(f.read_text())
            if t.get("steps"):
                rows.append(
                    {
                        "scenario": t.get("task_id"),
                        "text": steps_to_actions_text(t["steps"]),
                        "label": label,
                    }
                )
    return rows


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
                    "text": steps_to_actions_text(steps),
                    "label": int(median_score(js) >= 3),
                }
            )
    return rows


@torch.no_grad()
def score_all(model, tok, rows, max_length, batch, device) -> list[float]:
    model.eval()
    out = []
    for i in range(0, len(rows), batch):
        enc = tok(
            [r["text"] for r in rows[i : i + batch]],
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
            padding=True,
        ).to(device)
        out += torch.sigmoid(model(**enc)).float().cpu().tolist()
    return out


def report(name: str, scores, rows, held_out, flag_rate: float) -> dict:
    labels = [r["label"] for r in rows]
    k = int(round(flag_rate * len(scores)))
    order = sorted(range(len(scores)), key=lambda i: -scores[i])
    flagged = set(order[:k])
    tp = sum(labels[i] for i in flagged)
    fp, fn = k - tp, sum(labels) - tp
    tn = len(scores) - tp - fp - fn
    ho = [i for i, r in enumerate(rows) if r["scenario"] in held_out]
    block = {
        "auroc": auroc(scores, labels),
        "auroc_held_out_scenarios": (
            auroc([scores[i] for i in ho], [labels[i] for i in ho]) if ho else None
        ),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": tp / max(1, tp + fp),
        "recall": tp / max(1, tp + fn),
        "accuracy": (tp + tn) / len(scores),
        "per_arm_auroc": {
            arm: auroc(
                [s for s, r in zip(scores, rows) if r["arm"] == arm],
                [r["label"] for r in rows if r["arm"] == arm],
            )
            for arm in ARMS
        },
    }
    print(
        f"{name:18} AUROC {block['auroc']:.3f} | held-out {block['auroc_held_out_scenarios']:.3f} "
        f"| precision {100 * block['precision']:.0f}% recall {100 * block['recall']:.0f}% acc {100 * block['accuracy']:.0f}% "
        f"| TP {tp} FP {fp} FN {fn} TN {tn}",
        flush=True,
    )
    return block


def train_one(freeze: bool, a, train_rows, eval_rows, held_out, device) -> dict:
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModel, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(a.encoder)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    enc = AutoModel.from_pretrained(a.encoder, dtype=torch.bfloat16)
    if freeze:
        for p in enc.parameters():
            p.requires_grad_(False)
    else:
        enc = get_peft_model(
            enc,
            LoraConfig(
                r=a.lora_r,
                lora_alpha=2 * a.lora_r,
                lora_dropout=0.05,
                target_modules=[
                    "q_proj",
                    "k_proj",
                    "v_proj",
                    "o_proj",
                    "gate_proj",
                    "up_proj",
                    "down_proj",
                ],
                task_type="FEATURE_EXTRACTION",
            ),
        )
    model = ConstraintModel(
        enc,
        enc.config.hidden_size if freeze else enc.base_model.model.config.hidden_size,
    ).to(device)
    model.head.float()
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(
        f"\n=== {'frozen encoder' if freeze else 'LoRA-tuned encoder'}: {trainable:,} trainable parameters",
        flush=True,
    )

    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=a.lr, weight_decay=0.01
    )
    lossf = nn.BCEWithLogitsLoss()
    rng = random.Random(0)
    order = list(range(len(train_rows)))
    step = 0
    for epoch in range(a.epochs):
        rng.shuffle(order)
        model.train()
        for i in range(0, len(order), a.batch):
            idx = order[i : i + a.batch]
            enc_in = tok(
                [train_rows[j]["text"] for j in idx],
                return_tensors="pt",
                truncation=True,
                max_length=a.max_length,
                padding=True,
            ).to(device)
            y = torch.tensor(
                [float(train_rows[j]["label"]) for j in idx], device=device
            )
            loss = lossf(model(**enc_in), y)
            loss.backward()
            nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0
            )
            opt.step()
            opt.zero_grad()
            step += 1
            if step % 20 == 0:
                print(
                    f"   epoch {epoch} step {step} loss {float(loss):.4f}", flush=True
                )

    scores = score_all(model, tok, eval_rows, a.max_length, a.eval_batch, device)
    name = "frozen" if freeze else "lora_finetuned"
    block = report(name, scores, eval_rows, held_out, a.flag_rate)
    block["trainable_parameters"] = trainable
    del model, enc
    torch.cuda.empty_cache()
    return block


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--expert-dir", type=Path, required=True)
    ap.add_argument("--unsafe-dir", type=Path, required=True)
    ap.add_argument("--eval-root", type=Path, required=True)
    ap.add_argument("--splits", type=Path, required=True)
    ap.add_argument("--encoder", default="Qwen/Qwen2.5-1.5B")
    ap.add_argument("--max-length", type=int, default=2048)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--eval-batch", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--flag-rate", type=float, default=0.46)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    held_out = set()
    sp = json.loads(a.splits.read_text())
    for lab in sp.get("labels", {}).values():
        held_out |= set(lab.get("held_out_task_ids", []))

    train_rows = [
        r
        for r in load_training(a.expert_dir, a.unsafe_dir)
        if r["scenario"] not in held_out
    ]
    eval_rows = load_eval(a.eval_root)
    print(
        f"train {len(train_rows)} ({sum(r['label'] for r in train_rows)} unsafe), "
        f"eval {len(eval_rows)} ({sum(r['label'] for r in eval_rows)} violations), "
        f"{len(held_out)} held-out scenarios",
        flush=True,
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"

    out = {"n_train": len(train_rows), "n_eval": len(eval_rows), "variants": {}}
    out["variants"]["frozen"] = train_one(
        True, a, train_rows, eval_rows, held_out, device
    )
    out["variants"]["lora_finetuned"] = train_one(
        False, a, train_rows, eval_rows, held_out, device
    )

    f, l = out["variants"]["frozen"], out["variants"]["lora_finetuned"]
    print(
        f"\nunfreezing the encoder moved AUROC {f['auroc']:.3f} -> {l['auroc']:.3f} "
        f"and precision {100 * f['precision']:.0f}% -> {100 * l['precision']:.0f}%",
        flush=True,
    )
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=2))
    print("wrote", a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
