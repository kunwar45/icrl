#!/usr/bin/env python
# ABOUTME: Plots the constraint model's stress test: how much of its headline score survives controls, whether it grades severity, and what it costs in false positives
# ABOUTME: Run: python scratch/plot_constraint_stress.py --stress output/odcv_constraint_stress.json --out output/plots/2026-09-08_constraint_stress.png
import argparse, json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

INK, MUTED, GRID, SURF = "#1A1F27", "#5B6470", "#E4E8ED", "#FFFFFF"
BLUE, RED, GREY, GREEN = "#1f5fbf", "#b3261e", "#9AA4B0", "#2e7d5b"

ap = argparse.ArgumentParser(); ap.add_argument("--stress", type=Path, required=True); ap.add_argument("--out", type=Path, required=True)
a = ap.parse_args(); d = json.loads(a.stress.read_text())

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14.6, 4.7), dpi=170)
fig.patch.set_facecolor(SURF)
for ax in (ax1, ax2, ax3):
    ax.set_facecolor(SURF); ax.yaxis.grid(True, color=GRID, lw=0.9, zorder=0); ax.set_axisbelow(True)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    for s in ("bottom", "left"): ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED, length=0, labelsize=9)

# 1: what survives the controls
labels = ["headline\n(pooled)", "same length\nband", "within one\nscenario", "length alone\n(baseline)"]
vals = [d["auroc_pooled"], d["length_matched"]["concordance"], d["within_cell"]["concordance"], d["auroc_of_length_alone"]]
cols = [BLUE, BLUE, RED, GREY]
ax1.bar(range(4), vals, 0.62, color=cols, zorder=3)
for i, v in enumerate(vals): ax1.text(i, v + 0.015, f"{v:.2f}", ha="center", fontsize=10, color=INK)
ax1.axhline(0.5, color=INK, lw=1, ls="--", zorder=4); ax1.text(3.42, 0.515, "chance", fontsize=8.5, color=INK, ha="right")
ax1.set_xticks(range(4)); ax1.set_xticklabels(labels, fontsize=9)
ax1.set_ylim(0, 1.0); ax1.set_ylabel("ranks a violation above a clean run", color=MUTED, fontsize=9.5)
ax1.set_title("Most of the score survives length —\nbut not the within-scenario test", loc="left", fontsize=11, color=INK)

# 2: severity ordering
sev = {float(k): v for k, v in d["by_severity"].items() if v["n"] >= 5}
xs = sorted(sev); ys = [sev[x]["mean_C"] for x in xs]
ax2.plot(xs, ys, marker="o", color=BLUE, lw=2, ms=7, zorder=3)
for x, y in zip(xs, ys): ax2.text(x, y + 0.03, f"{y:.2f}", ha="center", fontsize=9, color=INK)
ax2.axvline(3, color=RED, lw=1.2, ls="--", zorder=2)
ax2.text(3.05, 0.12, "the line the judges\ncall a violation", fontsize=8.5, color=RED, linespacing=1.4)
ax2.set_ylim(0, 1.12); ax2.set_xlabel("judge severity, 0 to 5", color=MUTED, fontsize=9.5)
ax2.set_ylabel("mean constraint score", color=MUTED, fontsize=9.5)
ax2.set_title("It is not graded: flat from severity 1 to 4", loc="left", fontsize=11, color=INK)

# 3: what it catches and what it costs
b = d["blind_spot"]
names = ["edit-shaped\nviolations", "report-shaped\nviolations", "clean runs\nwrongly flagged"]
vals3 = [b["edit_shaped"]["recall"], b["report_shaped"]["recall"], b["false_positive_rate"]]
ax3.bar(range(3), vals3, 0.6, color=[GREEN, GREEN, RED], zorder=3)
for i, v in enumerate(vals3): ax3.text(i, v + 0.02, f"{v:.0%}", ha="center", fontsize=10, color=INK)
ax3.set_xticks(range(3)); ax3.set_xticklabels(names, fontsize=9)
ax3.set_ylim(0, 1.08); ax3.set_ylabel("share", color=MUTED, fontsize=9.5)
ax3.set_title("Catches both kinds — at a heavy false-alarm cost", loc="left", fontsize=11, color=INK)

fig.suptitle("Stress-testing the constraint model on 625 traces from four policies it never trained on",
             x=0.006, ha="left", fontsize=13, color=INK)
fig.text(0.006, 0.015,
         f"Calibration error {d['ece']:.2f}, so the score is not a probability. On scenarios it never trained on it scores "
         f"{d['auroc_held_out_scenarios']:.2f}, no worse than on trained ones — the generalisation is real; the precision is not.",
         fontsize=8.8, color=MUTED)
fig.tight_layout(rect=(0, 0.045, 1, 0.93))
a.out.parent.mkdir(parents=True, exist_ok=True); fig.savefig(a.out, facecolor=SURF); print("saved", a.out)
