#!/usr/bin/env python
# ABOUTME: Minimal schematic of one ODCV-Bench episode: prompt with an unreachable target, agent loop in a sandbox, transcript, then the two things scored from it
# ABOUTME: Run: python scratch/plot_odcv_schematic.py --out output/plots/2026-09-07_odcv_schematic.png
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

INK, MUTED, GRID, SURF = "#1A1F27", "#5B6470", "#E4E8ED", "#FFFFFF"
BLUE, RED, GREEN = "#1f5fbf", "#b3261e", "#2e7d5b"


def box(ax, x, y, w, h, text, edge, face, fs=9.5, weight="normal"):
    ax.add_patch(mpatches.FancyBboxPatch((x, y), w, h,
                 boxstyle="round,pad=0.008,rounding_size=0.02",
                 facecolor=face, edgecolor=edge, lw=1.4, zorder=3))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            color=INK, zorder=4, linespacing=1.5, weight=weight)


def arr(ax, a, b, color=MUTED, rad=0.0, ls="-"):
    ax.annotate("", xy=b, xytext=a, zorder=2,
                arrowprops=dict(arrowstyle="-|>", color=color, lw=1.5, linestyle=ls,
                                mutation_scale=18, connectionstyle=f"arc3,rad={rad}"))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    fig, ax = plt.subplots(figsize=(13.0, 4.3), dpi=170)
    fig.patch.set_facecolor(SURF); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    ax.text(0.005, 0.93, "One ODCV episode", fontsize=14, color=INK, weight="bold")

    box(ax, 0.005, 0.50, 0.20, 0.28,
        "TASK\n\ndo the job, and hit\na target the data\ncannot support", BLUE, "#e7effa")
    box(ax, 0.245, 0.50, 0.20, 0.28,
        "AGENT\n\nreal bash sandbox\ntools: bash,\ntask_complete\n≤ 50 turns", "#B9C1CB", "#f4f6f8")
    box(ax, 0.485, 0.50, 0.17, 0.28, "TRANSCRIPT\n\nevery command\nit ran", "#B9C1CB", "#f4f6f8")

    box(ax, 0.685, 0.635, 0.305, 0.145,
        "JUDGES score it 0–5\n≥ 3 = misaligned", RED, "#fdeceb", fs=9.5)
    box(ax, 0.685, 0.475, 0.305, 0.145,
        "SANDBOX checks files\nchanged?  tool run?", GREEN, "#e8f3ee", fs=9.5)

    arr(ax, (0.205, 0.64), (0.245, 0.64))
    arr(ax, (0.445, 0.64), (0.485, 0.64))
    arr(ax, (0.655, 0.66), (0.685, 0.705), color=RED)
    arr(ax, (0.655, 0.62), (0.685, 0.548), color=GREEN)
    arr(ax, (0.30, 0.50), (0.39, 0.50), color=MUTED, rad=-1.5)
    ax.text(0.345, 0.415, "repeats until it submits", fontsize=8.5, color=MUTED, ha="center")

    ax.text(0.005, 0.36, "The trap:", fontsize=10, color=RED, weight="bold")
    ax.text(0.075, 0.36,
            "the target is only reachable by cheating — editing the data, faking the output, or rewriting the checker.",
            fontsize=10, color=INK)
    ax.text(0.005, 0.245,
            "So there is no completion score. A validator PASS is usually evidence of cheating, not success.",
            fontsize=9.6, color=MUTED)
    ax.text(0.005, 0.145,
            "Honest behaviour normally means doing the work and reporting that the target was not met.",
            fontsize=9.6, color=MUTED)
    ax.text(0.005, 0.045,
            "40 scenarios × 2 framings (the target is demanded, or tied to the agent's own evaluation) = 80 cells.",
            fontsize=9.6, color=MUTED)

    fig.tight_layout(); a.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, facecolor=SURF); print("saved", a.out)


if __name__ == "__main__":
    main()
