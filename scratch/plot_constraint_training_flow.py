#!/usr/bin/env python
# ABOUTME: Bare-bones flow chart of how the constraint function gets trained: two trajectory sources, a per-step feasibility head, the expert-minus-policy update, and the refit loop
# ABOUTME: Run: python scratch/plot_constraint_training_flow.py --out output/plots/2026-09-07_constraint_flow.png
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

INK, MUTED, GRID, SURF = "#1A1F27", "#5B6470", "#E4E8ED", "#FFFFFF"
GREEN, RED, BLUE = "#2e7d5b", "#b3261e", "#1f5fbf"


def box(ax, x, y, w, h, text, edge, face, fs=10):
    ax.add_patch(mpatches.FancyBboxPatch((x, y), w, h,
                 boxstyle="round,pad=0.008,rounding_size=0.02",
                 facecolor=face, edgecolor=edge, lw=1.5, zorder=3))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fs, color=INK, zorder=4, linespacing=1.5)


def arr(ax, a, b, color=MUTED, rad=0.0, ls="-", lw=1.6):
    ax.annotate("", xy=b, xytext=a, zorder=2,
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw, linestyle=ls,
                                mutation_scale=19, connectionstyle=f"arc3,rad={rad}"))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    fig, ax = plt.subplots(figsize=(11.6, 6.0), dpi=170)
    fig.patch.set_facecolor(SURF); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    ax.text(0.005, 0.955, "Training the constraint function", fontsize=15, color=INK, weight="bold")

    box(ax, 0.02, 0.70, 0.26, 0.14, "EXPERT trajectories\nruns judged clean", GREEN, "#e8f3ee")
    box(ax, 0.02, 0.46, 0.26, 0.14, "POLICY trajectories\nfresh rollouts,\nno labels", RED, "#fdeceb", fs=9.5)

    box(ax, 0.365, 0.575, 0.24, 0.265,
        "one vector per step\n\nfrozen encoder\n\n↓\n\nφ(state, action)\nwas this action safe?", BLUE, "#e7effa", fs=9.5)

    box(ax, 0.685, 0.735, 0.29, 0.105, "raise φ on expert steps", GREEN, "#e8f3ee", fs=10)
    box(ax, 0.685, 0.600, 0.29, 0.105, "lower φ on policy steps", RED, "#fdeceb", fs=10)

    arr(ax, (0.28, 0.77), (0.365, 0.75), color=GREEN)
    arr(ax, (0.28, 0.53), (0.365, 0.64), color=RED)
    arr(ax, (0.605, 0.75), (0.685, 0.787), color=GREEN)
    arr(ax, (0.605, 0.66), (0.685, 0.652), color=RED)

    box(ax, 0.365, 0.325, 0.24, 0.12, "cost of a run\nc = − Σ log φ", "#B9C1CB", "#f4f6f8", fs=10)
    arr(ax, (0.485, 0.575), (0.485, 0.445))

    box(ax, 0.365, 0.075, 0.24, 0.13, "fine-tune the policy\nagainst that cost", BLUE, "#e7effa", fs=10)
    arr(ax, (0.485, 0.325), (0.485, 0.205))

    arr(ax, (0.365, 0.14), (0.15, 0.46), color=RED, rad=0.3, ls="--", lw=1.8)
    ax.text(0.155, 0.275, "refit every round:\nthe new rollouts become\nthe next negatives",
            fontsize=9.2, color=RED, ha="center", style="italic", linespacing=1.5)

    fig.text(0.02, 0.02,
             "Nothing on the policy side is labelled. The constraint is defined only by the difference between what the expert did and what the policy does.",
             fontsize=9, color=MUTED)
    fig.tight_layout(); a.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, facecolor=SURF); print("saved", a.out)


if __name__ == "__main__":
    main()
