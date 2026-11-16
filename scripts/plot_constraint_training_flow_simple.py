#!/usr/bin/env python
# ABOUTME: One-slide, five-step flow of how the ODCV constraint function is trained, sized so every box fits its text
# ABOUTME: Run: python scripts/plot_constraint_training_flow_simple.py
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

S, INK, INK2, MUTED, LINE = "#fcfcfb", "#1c1b19", "#57534e", "#8a857e", "#d5d1cb"
BLUE, BLUE_BG = "#2563eb", "#eaf1fd"

# Steps 1-3 collect and label data; 4-5 are the model. Colour encodes that split.
STEPS = [
    ("1", "Sample one policy",   "Same prompt, temp 0.7,\nsampled many times over.",           MUTED),
    ("2", "Label what it did",   "An LLM judge scores it 0–5.\nA second auditor checks.\nKeep only where they agree.", MUTED),
    ("3", "Two clean piles",     "232 expert  (severity 0–1)\n294 unsafe  (severity 3–5)\n+ 592 on-policy rollouts",   MUTED),
    ("4", "Train the classifier","Scenario + the agent's\nthoughts + its actions, read\nby Qwen2.5-7B with LoRA.",     BLUE),
    ("5", "Score a trajectory",  "C near 0  →  safe\nC near 1  →  violation",                          BLUE),
]

FIG_W, FIG_H = 15.0, 5.4
fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
fig.patch.set_facecolor(S); ax.set_facecolor(S)
ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

ax.text(.5, .965, "How the constraint function is trained",
        color=INK, fontsize=20, fontweight="bold", ha="center", va="top")

M, GAP = .018, .024
W = (1 - 2 * M - 4 * GAP) / 5
TOP, BOT = .795, .445                     # box height fits three body lines

for i, (num, head, body, c) in enumerate(STEPS):
    x = M + i * (W + GAP)
    ax.add_patch(FancyBboxPatch((x, BOT), W, TOP - BOT,
        boxstyle="round,pad=0,rounding_size=.010",
        facecolor=BLUE_BG if c is BLUE else S,
        edgecolor=c if c is BLUE else LINE, linewidth=1.6, zorder=2))
    ax.text(x + .014, TOP - .035, num, color=c, fontsize=15,
            fontweight="bold", va="center", ha="left")
    ax.text(x + .040, TOP - .035, head, color=INK, fontsize=12.5,
            fontweight="bold", va="center", ha="left")
    ax.text(x + .014, TOP - .082, body, color=INK2, fontsize=10.8,
            va="top", ha="left", linespacing=1.85)
    if i < 4:
        ax.add_patch(FancyArrowPatch((x + W + .004, (TOP + BOT) / 2),
            (x + W + GAP - .004, (TOP + BOT) / 2), arrowstyle="-|>",
            mutation_scale=15, color=MUTED, lw=1.6))

# ---- result strip --------------------------------------------------------
ax.add_patch(FancyBboxPatch((M, .075), 1 - 2 * M, .265,
    boxstyle="round,pad=0,rounding_size=.010",
    facecolor=BLUE_BG, edgecolor=BLUE, linewidth=1.8))
ax.text(M + .022, .275, "F1 0.847", color=BLUE, fontsize=30,
        fontweight="bold", va="center")
ax.text(M + .022, .155, "on 10 scenarios it never saw", color=INK2,
        fontsize=11.5, va="center")
ax.text(M + .245, .275, "0.828", color=INK, fontsize=30,
        fontweight="bold", va="center")
ax.text(M + .245, .155, "what two 70B judges score against each other",
        color=INK2, fontsize=11.5, va="center")
ax.plot([M + .585, M + .585], [.115, .300], color=BLUE, lw=1.2, alpha=.45)
ax.text(M + .615, .215, "So it is as reliable as the judge\nlabel itself is well defined.",
        color=BLUE, fontsize=15, fontweight="bold", va="center", linespacing=1.6)

fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
out = Path("output/plots/constraint_training_flow_simple.png")
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, dpi=200, facecolor=S)
print(f"wrote {out}")
