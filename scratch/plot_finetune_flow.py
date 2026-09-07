#!/usr/bin/env python
# ABOUTME: Bare-bones flow chart of the constrained fine-tuning round: serve, sample, score each rollout twice, sort into desirable and undesirable, update the policy, move the multiplier
# ABOUTME: Run: python scratch/plot_finetune_flow.py --out output/plots/2026-09-07_finetune_flow.png
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

INK, MUTED, GRID, SURF = "#1A1F27", "#5B6470", "#E4E8ED", "#FFFFFF"
GREEN, RED, BLUE, GREY = "#2e7d5b", "#b3261e", "#1f5fbf", "#B9C1CB"


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
    fig, ax = plt.subplots(figsize=(12.6, 6.2), dpi=170)
    fig.patch.set_facecolor(SURF); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    ax.text(0.005, 0.955, "One fine-tuning round", fontsize=15, color=INK, weight="bold")
    ax.text(0.005, 0.905, "No judge is involved anywhere in this loop.", fontsize=9.6, color=MUTED)

    box(ax, 0.02, 0.66, 0.20, 0.16, "SAMPLE\n\n6 rollouts per cell\nfrom the current\npolicy", BLUE, "#e7effa", fs=9.5)
    box(ax, 0.28, 0.66, 0.235, 0.16,
        "SCORE each one twice\n\nR : did it run the tool\n     and submit?\nC : constraint model", GREY, "#f4f6f8", fs=9.5)

    box(ax, 0.60, 0.755, 0.375, 0.09, "DESIRABLE   did the work, low C", GREEN, "#e8f3ee", fs=10)
    box(ax, 0.60, 0.645, 0.375, 0.09, "UNDESIRABLE   did the work, high C", RED, "#fdeceb", fs=10)
    ax.text(0.7875, 0.60, "the uncertain middle is dropped", fontsize=8.8, color=MUTED, ha="center")

    arr(ax, (0.22, 0.74), (0.28, 0.74), color=BLUE)
    arr(ax, (0.515, 0.755), (0.60, 0.80), color=GREEN)
    arr(ax, (0.515, 0.725), (0.60, 0.69), color=RED)

    box(ax, 0.315, 0.325, 0.37, 0.165,
        "UPDATE the policy\n\nraise the likelihood of desirable runs,\nlower it for undesirable ones,\nweighted by λ, anchored to the start", BLUE, "#e7effa", fs=9.5)
    arr(ax, (0.7875, 0.645), (0.60, 0.49), color=RED, rad=0.15)
    arr(ax, (0.7875, 0.755), (0.62, 0.492), color=GREEN, rad=0.28)

    box(ax, 0.315, 0.09, 0.37, 0.135,
        "MOVE λ\n\nconstraint above budget → λ rises\nso violations push harder next round", GREY, "#f4f6f8", fs=9.5)
    arr(ax, (0.50, 0.325), (0.50, 0.225))

    arr(ax, (0.315, 0.157), (0.12, 0.66), color=BLUE, rad=0.32, ls="--", lw=1.8)
    ax.text(0.145, 0.40, "next round", fontsize=9.2, color=BLUE, ha="center", style="italic")

    fig.text(0.02, 0.02,
             "Both classes must have done the work, so the policy learns how the task was done and never learns to simply stop early.",
             fontsize=9, color=MUTED)
    fig.tight_layout(); a.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, facecolor=SURF); print("saved", a.out)


if __name__ == "__main__":
    main()
