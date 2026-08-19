# ABOUTME: Trivial baselines on the SAME held-out split the constraint gate uses
# ABOUTME: Run: python scratch/gate_baselines.py <split_dir>   (prints AUROC to beat)
"""
An AUROC means nothing on its own. If a single scalar already separates the
held-out classes, a constraint head reporting the same number has demonstrated
nothing — it may simply have rediscovered that scalar.

These are the numbers C_theta has to beat to be evidence of anything.
"""
import json, sys
from pathlib import Path

def auroc(scores, labels):
    """Rank-based AUROC; labels 1 = unsafe."""
    pairs = sorted(zip(scores, labels))
    n1 = sum(labels); n0 = len(labels) - n1
    if n0 == 0 or n1 == 0: return float("nan")
    rank, i, ranks = 1, 0, {}
    # average ranks over ties
    vals = [p[0] for p in pairs]
    r = [0.0] * len(pairs)
    i = 0
    while i < len(pairs):
        j = i
        while j + 1 < len(pairs) and vals[j + 1] == vals[i]:
            j += 1
        avg = (i + j) / 2.0 + 1
        for k in range(i, j + 1):
            r[k] = avg
        i = j + 1
    s1 = sum(r[k] for k in range(len(pairs)) if pairs[k][1] == 1)
    return (s1 - n1 * (n1 + 1) / 2) / (n0 * n1)

def load(path):
    return [json.loads(l) for l in open(path)]

split = Path(sys.argv[1])
safe = load(split / "eval" / "safe_held_out.jsonl")
unsafe = load(split / "eval" / "unsafe_held_out.jsonl")
print(f"held-out: {len(safe)} safe, {len(unsafe)} unsafe")

rows = [(t, 0) for t in safe] + [(t, 1) for t in unsafe]
labels = [l for _, l in rows]

def feat(t, name):
    if name == "n_steps":
        return t.get("n_steps") or len(t.get("steps") or [])
    if name == "reward":
        return t.get("reward") or 0.0
    if name == "has_break":
        return 1.0 if any("break" in (s.get("action") or "") for s in t.get("steps") or []) else 0.0
    if name == "has_noop":
        return 1.0 if any("noop" in (s.get("action") or "") for s in t.get("steps") or []) else 0.0
    return 0.0

print("\nBASELINES on the held-out split (AUROC, 1 = unsafe):")
for name in ("n_steps", "reward", "has_break", "has_noop"):
    a = auroc([feat(t, name) for t, _ in rows], labels)
    flag = "  <-- C_theta must beat this" if name == "n_steps" else ""
    print(f"  {name:12s} {a:.4f}{flag}")

import statistics as st
for label, name in ((0, "safe"), (1, "unsafe")):
    xs = [feat(t, "n_steps") for t, l in rows if l == label]
    print(f"  {name:6s} n_steps: n={len(xs)} med={st.median(xs):.1f} mean={sum(xs)/len(xs):.2f}")
