#!/usr/bin/env python
# ABOUTME: Plots the KTO rounds' implied rewards and log-probabilities to show likelihood collapse: the margin improves while both classes are pushed down
# ABOUTME: Run: python scratch/plot_kto_training_pathology.py --mirror <dir with round_*/train_metrics.json> --out output/plots/2026-09-08_kto_training_pathology.png
import argparse, json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

INK, MUTED, GRID, SURF = "#1A1F27", "#5B6470", "#E4E8ED", "#FFFFFF"
GREEN, RED, BLUE = "#2e7d5b", "#b3261e", "#1f5fbf"

ap = argparse.ArgumentParser(); ap.add_argument("--mirror", type=Path, required=True); ap.add_argument("--out", type=Path, required=True)
a = ap.parse_args()
rounds = sorted(p for p in a.mirror.glob("round_*/train_metrics.json"))
fig, axes = plt.subplots(1, len(rounds), figsize=(3.5 * len(rounds), 4.4), dpi=170, sharey=False)
fig.patch.set_facecolor(SURF)
for ax, p in zip(axes, rounds):
    d = json.loads(p.read_text())
    lh = [x for x in d["log_history"] if "rewards/chosen" in x]
    xs = list(range(len(lh)))
    ax.set_facecolor(SURF)
    ax.axhline(0, color=INK, lw=1, ls="--", zorder=2)
    ax.plot(xs, [x["rewards/chosen"] for x in lh], color=GREEN, lw=2, label="desirable", zorder=3)
    ax.plot(xs, [x["rewards/rejected"] for x in lh], color=RED, lw=2, label="undesirable", zorder=3)
    ax.set_title(f"round {p.parent.name.split('_')[1]}  ·  {d['n_rows']} rows", fontsize=10.5, color=INK, loc="left")
    ax.set_xlabel("optimiser step", color=MUTED, fontsize=9.5)
    ax.yaxis.grid(True, color=GRID, lw=0.9, zorder=0); ax.set_axisbelow(True)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    for s in ("bottom", "left"): ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED, length=0, labelsize=9)
axes[0].set_ylabel("implied reward  (log-ratio to the starting policy)", color=MUTED, fontsize=9.5)
axes[0].legend(frameon=False, fontsize=9.5, loc="lower left")
fig.suptitle("Both classes are pushed below the starting policy, so the margin improves by suppression",
             x=0.006, ha="left", fontsize=13, color=INK)
fig.text(0.006, 0.015,
         "Above the dashed line the policy finds a run MORE likely than the model it started from; below it, less likely. "
         "The desirable runs should rise. They fall.",
         fontsize=8.8, color=MUTED)
fig.tight_layout(rect=(0, 0.045, 1, 0.93))
a.out.parent.mkdir(parents=True, exist_ok=True); fig.savefig(a.out, facecolor=SURF); print("saved", a.out)
