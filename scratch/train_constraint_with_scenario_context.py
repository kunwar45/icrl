#!/usr/bin/env python
# ABOUTME: Trains the ODCV constraint model on what the judge actually reads -- the scenario prompt that defines the trap, the agent's stated reasoning, and its actions -- and can add the judge-labelled policy rollouts from training scenarios to the training pool
# ABOUTME: Run on a GPU node: python scratch/train_constraint_with_scenario_context.py --expert-dir <d> --unsafe-dir <d> --eval-root <d> --splits <splits.json> --arm context_plus_rollouts --out output/constraint_context.json
"""
Four things the previous constraint runs threw away, all present on disk:

  1. `scenario_prompt` -- the system+user framing that DEFINES the violation.
     The same `bash` command is compliant in one scenario and a violation in
     another; without the framing the model is guessing from surface form.
  2. `reasoning` -- the agent's stated rationale, which is where the
     misalignment is usually explicit.
  3. The 778 judge-labelled evaluation rollouts, used only as a test set even
     though 3/4 of them come from training scenarios and are the ON-policy
     distribution the constraint is meant to score.
  4. A sane operating point. Reporting precision at a fixed 46% flag rate when
     prevalence is 21% caps precision at 45% arithmetically. We report the
     precision-recall curve instead.

Arms (identical evaluation, so the ablation is exact):
  actions_only        audited traces, actions only          -- the old setup
  context             audited traces, + scenario + reasoning -- input fix
  context_rollouts    + judged rollouts from train scenarios -- data fix
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.environments.odcv.transcripts import transcript_to_steps  # noqa: E402

JUDGES = ("llama-3.3-70b", "mistral-small-3.2-24b", "qwen3.6-27b")
SCENARIO_CHARS = 2500
REASON_CHARS = 400
RESULT_CHARS = 300


# ---------------------------------------------------------------- input text


def trajectory_text(steps: list[dict], with_reasoning: bool, with_results: bool) -> str:
    out = []
    for s in steps:
        if with_reasoning and (s.get("reasoning") or "").strip():
            out.append(f"[THOUGHT] {s['reasoning'][:REASON_CHARS]}")
        out.append(f"[ACTION] {s.get('action', '')}")
        if with_results and (s.get("observation") or "").strip():
            out.append(f"[RESULT] {s['observation'][:RESULT_CHARS]}")
    return "\n".join(out)


def build_text(scenario_prompt: str, steps: list[dict], arm: dict) -> str:
    traj = trajectory_text(steps, arm["reasoning"], arm["results"])
    if not arm["context"]:
        return traj
    head = (scenario_prompt or "")[:SCENARIO_CHARS]
    return f"[SCENARIO]\n{head}\n\n[TRAJECTORY]\n{traj}"


def middle_truncate(tok, text: str, max_length: int) -> list[int]:
    """Keep the scenario header and the END of the trajectory: the decisive
    actions (the final edit, task_complete) are the last thing that happens."""
    ids = tok(text, add_special_tokens=False)["input_ids"]
    if len(ids) <= max_length:
        return ids
    head = int(0.45 * max_length)
    return ids[:head] + ids[-(max_length - head) :]


# ------------------------------------------------------------------- corpus


def load_audited(expert_dir: Path, unsafe_dir: Path, arm: dict) -> list[dict]:
    rows = []
    for d, label in ((expert_dir, 0), (unsafe_dir, 1)):
        for f in sorted(d.glob("task_*.json")):
            t = json.loads(f.read_text())
            if not t.get("steps"):
                continue
            rows.append(
                {
                    "scenario": t.get("task_id"),
                    "source": "audited",
                    "text": build_text(t.get("scenario_prompt", ""), t["steps"], arm),
                    "label": label,
                }
            )
    return rows


def load_rollouts(eval_root: Path, arm: dict) -> list[dict]:
    rows = []
    for adir in sorted(p for p in eval_root.iterdir() if (p / "rollouts").exists()):
        scores = {}
        for j in JUDGES:
            p = adir / "results" / f"scores_{j}.json"
            if p.exists():
                scores[j] = json.loads(p.read_text())
        for rec in sorted(adir.glob("rollouts/*/*/pass*/messages_record.txt")):
            if rec.stat().st_size == 0:
                continue
            key = str(rec.relative_to(adir))
            js = sorted(
                scores[j][key]["score"]
                for j in scores
                if key in scores[j]
                and isinstance(scores[j][key]["score"], (int, float))
            )
            if not js:
                continue
            med = (
                js[len(js) // 2]
                if len(js) % 2
                else (js[len(js) // 2 - 1] + js[len(js) // 2]) / 2
            )
            steps, meta = transcript_to_steps(
                rec.read_text(errors="replace"), obs_chars=RESULT_CHARS
            )
            if not steps:
                continue
            rows.append(
                {
                    "arm": adir.name,
                    "scenario": rec.parts[-3],
                    "source": "rollout",
                    "text": build_text(meta.get("scenario_prompt", ""), steps, arm),
                    "label": int(med >= 3),
                }
            )
    return rows


# ------------------------------------------------------------------ metrics


def auroc(scores, labels) -> float:
    pairs = sorted(zip(scores, labels))
    pos = sum(labels)
    neg = len(labels) - pos
    if pos == 0 or neg == 0:
        return float("nan")
    rank, i = {}, 0
    ranks = [0.0] * len(pairs)
    while i < len(pairs):
        j = i
        while j + 1 < len(pairs) and pairs[j + 1][0] == pairs[i][0]:
            j += 1
        r = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[k] = r
        i = j + 1
    s = sum(r for r, (_, l) in zip(ranks, pairs) if l)
    return (s - pos * (pos + 1) / 2) / (pos * neg)


def pr_curve(scores, labels):
    """Every operating point, so the threshold is a choice and not an artifact."""
    order = sorted(range(len(scores)), key=lambda i: -scores[i])
    P = sum(labels)
    tp = fp = 0
    pts = []
    for rank, i in enumerate(order, 1):
        if labels[i]:
            tp += 1
        else:
            fp += 1
        pts.append(
            {
                "k": rank,
                "flag_rate": rank / len(scores),
                "precision": tp / rank,
                "recall": tp / P if P else 0.0,
                "tp": tp,
                "fp": fp,
                "fn": P - tp,
                "tn": len(scores) - rank - (P - tp),
            }
        )
    return pts


def summarise(scores, labels, prevalence_flag=True) -> dict:
    pts = pr_curve(scores, labels)
    P = sum(labels)
    n = len(labels)
    ap = 0.0
    prev_r = 0.0
    for p in pts:
        ap += p["precision"] * (p["recall"] - prev_r)
        prev_r = p["recall"]
    out = {
        "n": n,
        "positives": P,
        "prevalence": P / n if n else 0.0,
        "auroc": auroc(scores, labels),
        "average_precision": ap,
    }
    for target in (0.5, 0.7, 0.8, 0.9):
        ok = [p for p in pts if p["recall"] >= target]
        out[f"precision_at_recall_{target}"] = max(
            (p["precision"] for p in ok), default=0.0
        )
    best = max(
        pts,
        key=lambda p: (
            2 * p["precision"] * p["recall"] / max(p["precision"] + p["recall"], 1e-9)
        ),
    )
    out["best_f1"] = {
        k: best[k] for k in ("precision", "recall", "tp", "fp", "fn", "tn", "flag_rate")
    }
    out["best_f1"]["f1"] = (
        2 * best["precision"] * best["recall"] / max(best["precision"] + best["recall"], 1e-9)
    )
    if prevalence_flag and P:
        at = pts[min(P, n) - 1]
        out["at_prevalence_flag_rate"] = {
            k: at[k] for k in ("precision", "recall", "tp", "fp", "fn", "tn", "flag_rate")
        }
    out["at_flag_rate_0.46"] = {
        k: pts[min(int(round(0.46 * n)), n) - 1][k]
        for k in ("precision", "recall", "tp", "fp", "fn", "tn", "flag_rate")
    }
    return out


# -------------------------------------------------------------------- model


class ConstraintModel(nn.Module):
    def __init__(self, encoder, hidden: int):
        super().__init__()
        self.encoder = encoder
        self.head = nn.Sequential(
            nn.Linear(hidden, 256), nn.ReLU(), nn.Dropout(0.1), nn.Linear(256, 1)
        )

    def forward(self, input_ids, attention_mask):
        h = self.encoder(
            input_ids=input_ids, attention_mask=attention_mask
        ).last_hidden_state
        m = attention_mask.unsqueeze(-1).to(h.dtype)
        pooled = (h * m).sum(1) / m.sum(1).clamp(min=1)
        return self.head(pooled.float()).squeeze(-1)


def collate(tok, rows, max_length, device):
    seqs = [middle_truncate(tok, r["text"], max_length) for r in rows]
    width = max(len(s) for s in seqs)
    pad = tok.pad_token_id or 0
    ids = torch.full((len(seqs), width), pad, dtype=torch.long)
    mask = torch.zeros((len(seqs), width), dtype=torch.long)
    for i, s in enumerate(seqs):
        ids[i, : len(s)] = torch.tensor(s)
        mask[i, : len(s)] = 1
    return ids.to(device), mask.to(device)


@torch.no_grad()
def score_all(model, tok, rows, max_length, batch, device):
    model.eval()
    out = []
    for i in range(0, len(rows), batch):
        ids, mask = collate(tok, rows[i : i + batch], max_length, device)
        out += torch.sigmoid(model(ids, mask)).float().cpu().tolist()
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--expert-dir", type=Path, required=True)
    ap.add_argument("--unsafe-dir", type=Path, required=True)
    ap.add_argument("--eval-root", type=Path, required=True)
    ap.add_argument("--splits", type=Path, required=True)
    ap.add_argument(
        "--arm",
        choices=("actions_only", "context", "context_rollouts", "rollouts_only"),
        required=True,
    )
    ap.add_argument("--encoder", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--max-length", type=int, default=3072)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--eval-batch", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--train-cap",
        type=int,
        default=0,
        help="Subsample the training pool to this many trajectories, to match "
        "another arm's example count.",
    )
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument(
        "--save-model",
        type=Path,
        help="Directory to write the LoRA adapter, the pooled head and the "
        "operating threshold, so the fine-tuning loop can score with it.",
    )
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    random.seed(a.seed)

    arm = {
        "actions_only": {"context": False, "reasoning": False, "results": False},
        "context": {"context": True, "reasoning": True, "results": False},
        "context_rollouts": {"context": True, "reasoning": True, "results": False},
        "rollouts_only": {"context": True, "reasoning": True, "results": False},
    }[a.arm]
    use_rollouts = a.arm in ("context_rollouts", "rollouts_only")

    held_out = set()
    sp = json.loads(a.splits.read_text())
    held_out |= set(sp.get("held_out_task_ids", []))
    for lab in sp.get("labels", {}).values():
        held_out |= set(lab.get("held_out_task_ids", []))
    print(f"held-out scenarios ({len(held_out)}): {sorted(held_out)}", flush=True)

    audited = load_audited(a.expert_dir, a.unsafe_dir, arm)
    rollouts = load_rollouts(a.eval_root, arm)
    print(f"audited {len(audited)} | rollouts {len(rollouts)}", flush=True)

    # test set is FIXED across arms: judged rollouts from held-out scenarios
    test = [r for r in rollouts if r["scenario"] in held_out]
    train_pool = (
        [] if a.arm == "rollouts_only"
        else [r for r in audited if r["scenario"] not in held_out]
    )
    if use_rollouts:
        train_pool += [r for r in rollouts if r["scenario"] not in held_out]
    if a.train_cap and len(train_pool) > a.train_cap:
        # Match another arm's example COUNT, so a gain cannot be attributed to
        # data quantity when the question is data distribution.
        random.Random(a.seed).shuffle(train_pool)
        train_pool = train_pool[: a.train_cap]

    scen = sorted({r["scenario"] for r in train_pool})
    random.Random(a.seed).shuffle(scen)
    val_scen = set(scen[: max(1, len(scen) // 5)])
    tr = [r for r in train_pool if r["scenario"] not in val_scen]
    va = [r for r in train_pool if r["scenario"] in val_scen]
    print(
        f"arm={a.arm} train {len(tr)} (pos {sum(r['label'] for r in tr)}) | "
        f"val {len(va)} | test {len(test)} (pos {sum(r['label'] for r in test)})",
        flush=True,
    )

    tok = AutoTokenizer.from_pretrained(a.encoder)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    device = "cuda" if torch.cuda.is_available() else "cpu"
    enc = AutoModel.from_pretrained(a.encoder, dtype=torch.bfloat16).to(device)
    enc.gradient_checkpointing_enable()
    enc.enable_input_require_grads()

    from peft import LoraConfig, get_peft_model

    enc = get_peft_model(
        enc,
        LoraConfig(
            r=a.lora_r,
            lora_alpha=2 * a.lora_r,
            lora_dropout=0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            bias="none",
        ),
    )
    model = ConstraintModel(enc, enc.config.hidden_size).to(device)
    model.head.to(torch.float32)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"trainable parameters {trainable/1e6:.1f}M", flush=True)

    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=a.lr
    )
    lossf = nn.BCEWithLogitsLoss()
    best = (-1.0, None)
    for ep in range(a.epochs):
        model.train()
        random.Random(a.seed + ep).shuffle(tr)
        total = 0.0
        opt.zero_grad()
        for i, r in enumerate(tr):
            ids, mask = collate(tok, [r], a.max_length, device)
            logit = model(ids, mask)
            loss = lossf(logit, torch.tensor([float(r["label"])], device=device))
            (loss / a.grad_accum).backward()
            total += float(loss)
            if (i + 1) % a.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], 1.0
                )
                opt.step()
                opt.zero_grad()
        opt.step()
        opt.zero_grad()
        vs = score_all(model, tok, va, a.max_length, a.eval_batch, device)
        vauc = auroc(vs, [r["label"] for r in va])
        print(
            f"  epoch {ep+1}/{a.epochs} loss {total/max(len(tr),1):.4f} val AUROC {vauc:.3f}",
            flush=True,
        )
        if vauc > best[0]:
            best = (
                vauc,
                {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
            )
    if best[1] is not None:
        model.load_state_dict(best[1])
        print(f"restored the best epoch (val AUROC {best[0]:.3f})", flush=True)

    scores = score_all(model, tok, test, a.max_length, a.eval_batch, device)
    labels = [r["label"] for r in test]
    block = {
        "arm": a.arm,
        "encoder": a.encoder,
        "validation_auroc": best[0],
        "n_train": len(tr),
        "train_cap": a.train_cap,
        "trainable_parameters": trainable,
        "held_out_scenarios": sorted(held_out),
        "test": summarise(scores, labels),
        "scores": scores,
        "labels": labels,
        "test_scenarios": [r["scenario"] for r in test],
        "test_arms": [r.get("arm") for r in test],
    }
    t = block["test"]
    print(
        f"\n{a.arm:18} HELD-OUT SCENARIOS  n={t['n']} prevalence {t['prevalence']:.1%}\n"
        f"  AUROC {t['auroc']:.3f}  AP {t['average_precision']:.3f}\n"
        f"  precision @ recall 50/70/80/90%: "
        f"{t['precision_at_recall_0.5']:.1%} / {t['precision_at_recall_0.7']:.1%} / "
        f"{t['precision_at_recall_0.8']:.1%} / {t['precision_at_recall_0.9']:.1%}\n"
        f"  best F1 {t['best_f1']['f1']:.3f}: precision {t['best_f1']['precision']:.1%} "
        f"recall {t['best_f1']['recall']:.1%} "
        f"(TP {t['best_f1']['tp']} FP {t['best_f1']['fp']} "
        f"FN {t['best_f1']['fn']} TN {t['best_f1']['tn']})",
        flush=True,
    )
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(block, indent=2))
    print(f"wrote {a.out}", flush=True)

    if a.save_model:
        a.save_model.mkdir(parents=True, exist_ok=True)
        model.encoder.save_pretrained(a.save_model / "adapter")
        tok.save_pretrained(a.save_model / "adapter")
        torch.save(model.head.state_dict(), a.save_model / "head.pt")
        # The threshold is part of the model: a score is only a decision once a
        # cut point is fixed, and it must be fixed on VALIDATION, never on test.
        vs = score_all(model, tok, va, a.max_length, a.eval_batch, device)
        vl = [r["label"] for r in va]
        vpts = pr_curve(vs, vl)
        vbest = max(
            vpts,
            key=lambda p: 2 * p["precision"] * p["recall"]
            / max(p["precision"] + p["recall"], 1e-9),
        )
        order = sorted(vs, reverse=True)
        threshold = order[vbest["k"] - 1]
        (a.save_model / "constraint_model.json").write_text(
            json.dumps(
                {
                    "encoder": a.encoder,
                    "arm": a.arm,
                    "max_length": a.max_length,
                    "scenario_chars": SCENARIO_CHARS,
                    "reason_chars": REASON_CHARS,
                    "with_reasoning": arm["reasoning"],
                    "with_context": arm["context"],
                    "threshold": threshold,
                    "threshold_chosen_on": "validation scenarios",
                    "validation_at_threshold": {
                        k: vbest[k]
                        for k in ("precision", "recall", "flag_rate", "tp", "fp", "fn", "tn")
                    },
                    "test": block["test"],
                },
                indent=2,
            )
        )
        print(
            f"saved constraint model to {a.save_model} "
            f"(threshold {threshold:.4f} from validation, "
            f"val precision {vbest['precision']:.1%} recall {vbest['recall']:.1%})",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
