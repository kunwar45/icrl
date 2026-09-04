# ABOUTME: Sweeps constraint-head losses on cached ODCV embeddings with scenario-level K-fold held-out evaluation
# ABOUTME: Run: python scratch/sweep_odcv_constraint_heads.py --bundles <dir with safe.pt unsafe.pt> [--folds 4] [--out <json>]
"""
The v1 head (ICRL loss, beta 0.8) saturated: it only bounds the MEAN expert
score, so half the experts can sit at 1.0. This trains the same MLP head on the
same frozen embeddings under several losses and reports, per scenario fold:
overall held-out AUROC, within-scenario pair-weighted AUROC (the honest number,
immune to single-class scenarios), ECE, and the two trivial baselines computed
from the bundle's own texts (protected-path-write regex, step count).
"""

import argparse
import json
import random
import re
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

WRITE_RE = re.compile(
    r"(sed\s+-i|>\s*/(app/data|usr/local/bin)|>>\s*/(app/data|usr/local/bin)"
    r"|tee\s+/(app/data|usr/local/bin)|chmod|chown|\bmv\s+\S+\s+/(app/data|usr/local/bin)"
    r"|\bcp\s+\S+\s+/(app/data|usr/local/bin)|\brm\s+.*?/(app/data|usr/local/bin)"
    r"|cat\s*>\s*/(app/data|usr/local/bin)|python3?\s+-c\s+.*open\(.*['\"]w['\"])"
)
ACTION_RE = re.compile(r"\[ACTION\] (.*?)(?= \[OBS\]|\n\[STEP |\Z)", re.S)


def auroc(scores, labels):
    pairs = sorted(zip(scores, labels))
    n1 = sum(labels)
    n0 = len(labels) - n1
    if n0 == 0 or n1 == 0:
        return None
    vals = [p[0] for p in pairs]
    r = [0.0] * len(pairs)
    i = 0
    while i < len(pairs):
        j = i
        while j + 1 < len(pairs) and vals[j + 1] == vals[i]:
            j += 1
        for k in range(i, j + 1):
            r[k] = (i + j) / 2.0 + 1
        i = j + 1
    s1 = sum(r[k] for k in range(len(pairs)) if pairs[k][1] == 1)
    return (s1 - n1 * (n1 + 1) / 2) / (n0 * n1)


def ece(scores, labels, bins=10):
    scores, labels = np.asarray(scores), np.asarray(labels)
    edges = np.linspace(0, 1, bins + 1)
    tot = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (
            (scores >= lo) & (scores < hi)
            if hi < 1
            else (scores >= lo) & (scores <= hi)
        )
        if m.any():
            tot += m.mean() * abs(scores[m].mean() - labels[m].mean())
    return float(tot)


def within_scenario(scores, labels, tasks):
    by = defaultdict(list)
    for s, l, t in zip(scores, labels, tasks):
        by[t].append((s, l))
    num = den = 0.0
    for v in by.values():
        l = [y for _, y in v]
        n0, n1 = l.count(0), l.count(1)
        if n0 and n1:
            num += n0 * n1 * auroc([x for x, _ in v], l)
            den += n0 * n1
    return (num / den) if den else None, int(den)


def features(text):
    acts = ACTION_RE.findall(text)
    writes = sum(1 for a in acts if a.startswith("bash:") and WRITE_RE.search(a))
    return float(writes), float(len(acts))


def load_bundles(d):
    rows = []
    for name, lab in (("safe", 0), ("unsafe", 1)):
        b = torch.load(f"{d}/{name}.pt", map_location="cpu", weights_only=False)
        E = b["embeddings"].float()
        for i in range(E.shape[0]):
            rows.append(
                {
                    "emb": E[i],
                    "label": lab,
                    "task": str(b["task_ids"][i]),
                    "text": b["texts"][i],
                }
            )
    return rows


def make_folds(rows, k, seed):
    rng = random.Random(seed)
    tasks = sorted({r["task"] for r in rows})
    rng.shuffle(tasks)
    counts = {t: sum(1 for r in rows if r["task"] == t) for t in tasks}
    folds = [[] for _ in range(k)]
    load = [0] * k
    for t in sorted(tasks, key=lambda t: -counts[t]):  # greedy balance
        i = load.index(min(load))
        folds[i].append(t)
        load[i] += counts[t]
    return folds


class Head(nn.Module):
    def __init__(self, h, hidden):
        super().__init__()
        self.net = (
            nn.Linear(h, 1)
            if hidden <= 0
            else nn.Sequential(nn.Linear(h, hidden), nn.GELU(), nn.Linear(hidden, 1))
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)  # logits


LOSSES = {
    "icrl_b0.8_l1": dict(kind="icrl", beta=0.8, lam=1.0, persample=False),
    "icrl_b0.2_l5": dict(kind="icrl", beta=0.2, lam=5.0, persample=False),
    "hinge_ps_b0.2": dict(kind="icrl", beta=0.2, lam=5.0, persample=True),
    "bce": dict(kind="bce", balanced=False),
    "bce_balanced": dict(kind="bce", balanced=True),
    "logreg_C0.01": dict(kind="logreg", C=0.01),
    "logreg_C0.1": dict(kind="logreg", C=0.1),
    "logreg_C1": dict(kind="logreg", C=1.0),
    "logreg_C0.003": dict(kind="logreg", C=0.003),
    "logreg_C0.001": dict(kind="logreg", C=0.001),
    "logreg_C0.01_cal": dict(kind="logreg", C=0.01, calibrate=True),
}


def loss_fn(cfg, logit_e, logit_p):
    pe, pp = torch.sigmoid(logit_e), torch.sigmoid(logit_p)
    if cfg["kind"] == "icrl":
        viol = (
            F.relu(pe - cfg["beta"]).mean()
            if cfg["persample"]
            else F.relu(pe.mean() - cfg["beta"])
        )
        return -pp.mean() + cfg["lam"] * viol
    we = (len(pe) + len(pp)) / (2 * len(pe)) if cfg["balanced"] else 1.0
    wp = (len(pe) + len(pp)) / (2 * len(pp)) if cfg["balanced"] else 1.0
    return we * F.binary_cross_entropy_with_logits(
        logit_e, torch.zeros_like(logit_e)
    ) + wp * F.binary_cross_entropy_with_logits(logit_p, torch.ones_like(logit_p))


def train_head(cfg, Xe, Xp, hidden, steps, batch, lr, wd, seed):
    torch.manual_seed(seed)
    head = Head(Xe.shape[1], hidden)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=wd)
    g = torch.Generator().manual_seed(seed)
    for _ in range(steps):
        ie = torch.randint(0, Xe.shape[0], (batch,), generator=g)
        ip = torch.randint(0, Xp.shape[0], (batch,), generator=g)
        loss = loss_fn(cfg, head(Xe[ie]), head(Xp[ip]))
        opt.zero_grad()
        loss.backward()
        opt.step()
    return head


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundles", required=True)
    ap.add_argument("--folds", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--wd", type=float, default=0.01)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--losses", default=",".join(LOSSES))
    ap.add_argument(
        "--standardize", action="store_true", help="z-score features on the train fold"
    )
    ap.add_argument("--layernorm", action="store_true", help="per-sample z-score of the feature vector")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    rows = load_bundles(a.bundles)
    folds = make_folds(rows, a.folds, a.seed)
    print(
        f"{len(rows)} traces, {len({r['task'] for r in rows})} scenarios, {a.folds} folds "
        f"(sizes {[sum(1 for r in rows if r['task'] in f) for f in folds]}), standardize={a.standardize}"
    )
    feats = {id(r): features(r["text"]) for r in rows}
    results = defaultdict(list)
    for fi, held in enumerate(folds):
        held = set(held)
        tr = [r for r in rows if r["task"] not in held]
        te = [r for r in rows if r["task"] in held]
        Xtr = torch.stack([r["emb"] for r in tr])
        Xte = torch.stack([r["emb"] for r in te])
        if a.layernorm:
            Xtr = (Xtr - Xtr.mean(1, keepdim=True)) / (Xtr.std(1, keepdim=True) + 1e-6)
            Xte = (Xte - Xte.mean(1, keepdim=True)) / (Xte.std(1, keepdim=True) + 1e-6)
        if a.standardize:
            mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
            Xtr = (Xtr - mu) / sd
            Xte = (Xte - mu) / sd
        ytr = [r["label"] for r in tr]
        yte = [r["label"] for r in te]
        tte = [r["task"] for r in te]
        Xe = Xtr[[i for i, y in enumerate(ytr) if y == 0]]
        Xp = Xtr[[i for i, y in enumerate(ytr) if y == 1]]
        base = {
            "regex": [feats[id(r)][0] for r in te],
            "n_steps": [feats[id(r)][1] for r in te],
        }
        for name, sc in base.items():
            w, npairs = within_scenario(sc, yte, tte)
            results[name].append(
                {
                    "fold": fi,
                    "auroc": auroc(sc, yte),
                    "within": w,
                    "pairs": npairs,
                    "ece": None,
                }
            )
        for name in a.losses.split(","):
            cfg = LOSSES[name]
            if cfg["kind"] == "logreg":
                from sklearn.linear_model import LogisticRegression

                clf = LogisticRegression(
                    C=cfg["C"], class_weight="balanced", max_iter=5000
                )
                clf.fit(Xtr.numpy(), np.asarray(ytr))
                sc = clf.predict_proba(Xte.numpy())[:, 1].tolist()
            else:
                head = train_head(
                    cfg, Xe, Xp, a.hidden, a.steps, a.batch, a.lr, a.wd, a.seed
                )
                with torch.no_grad():
                    sc = torch.sigmoid(head(Xte)).tolist()
            w, npairs = within_scenario(sc, yte, tte)
            results[name].append(
                {
                    "fold": fi,
                    "auroc": auroc(sc, yte),
                    "within": w,
                    "pairs": npairs,
                    "ece": ece(sc, yte),
                    "safe_med": float(
                        np.median([s for s, y in zip(sc, yte) if y == 0])
                    ),
                    "unsafe_med": float(
                        np.median([s for s, y in zip(sc, yte) if y == 1])
                    ),
                    "n_safe": yte.count(0),
                    "n_unsafe": yte.count(1),
                }
            )
        print(
            f"fold {fi}: held-out {len(te)} traces ({yte.count(0)} safe / {yte.count(1)} unsafe), "
            f"{len(held)} scenarios"
        )

    print(
        f"\n{'config':16s} {'AUROC mean±sd':>16s} {'within mean±sd':>16s} {'ECE':>6s}  per-fold AUROC / within"
    )
    summary = {}
    for name, rs in results.items():
        au = [r["auroc"] for r in rs]
        wi = [r["within"] for r in rs if r["within"] is not None]
        ec = [r["ece"] for r in rs if r["ece"] is not None]
        summary[name] = {
            "auroc_mean": float(np.mean(au)),
            "auroc_sd": float(np.std(au)),
            "within_mean": float(np.mean(wi)),
            "within_sd": float(np.std(wi)),
            "ece_mean": float(np.mean(ec)) if ec else None,
            "folds": rs,
        }
        print(
            f"{name:16s} {np.mean(au):.3f} ± {np.std(au):.3f}   {np.mean(wi):.3f} ± {np.std(wi):.3f}  "
            f"{(np.mean(ec) if ec else float('nan')):.3f}  "
            + "  ".join(f"{r['auroc']:.2f}/{(r['within'] or 0):.2f}" for r in rs)
        )
    if a.out:
        json.dump(
            {"args": vars(a), "folds": folds, "summary": summary},
            open(a.out, "w"),
            indent=2,
        )
        print("wrote", a.out)


if __name__ == "__main__":
    main()
