#!/usr/bin/env python
# ABOUTME: Draws two explanatory figures: how the constraint function is trained (the ICRL alternating loop) and how it behaves on real traces (one score per trace, coloured by the judges' verdict)
# ABOUTME: Run: python scripts/plot_constraint_diagrams.py --transfer output/odcv_constraint_transfer.json --out-dir output/plots --date 2026-09-07
"""
Two deliberately simple pictures.

`*_constraint_training.png`  the training loop: where the expert trajectories and
the policy trajectories come from, what the feasibility function does to each,
and why the loop has to close (the nominal half is resampled from the current
policy every round, which is the step that makes this inverse constrained RL
rather than a one-off classifier).

`*_constraint_behaviour.png`  the behaviour: every evaluated trace as one point on
the score axis, split by what the judges said, so the question "can it tell a
safe trace from an unsafe one" is answered by looking at the overlap.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

INK, MUTED, GRID, SURF = "#1A1F27", "#5B6470", "#E4E8ED", "#FFFFFF"
SAFE, VIOLATION, ACCENT = "#2e7d5b", "#b3261e", "#1f5fbf"


def box(
    ax, x, y, w, h, text, face, edge, fontsize=9.5, weight="normal", text_color=INK
):
    ax.add_patch(
        mpatches.FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.012,rounding_size=0.02",
            facecolor=face,
            edgecolor=edge,
            linewidth=1.4,
            zorder=3,
        )
    )
    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        color=text_color,
        zorder=4,
        weight=weight,
        linespacing=1.45,
    )


def arrow(ax, xy_from, xy_to, color=MUTED, style="-|>", rad=0.0, lw=1.5, ls="-"):
    ax.annotate(
        "",
        xy=xy_to,
        xytext=xy_from,
        zorder=2,
        arrowprops=dict(
            arrowstyle=style,
            color=color,
            lw=lw,
            linestyle=ls,
            connectionstyle=f"arc3,rad={rad}",
            shrinkA=2,
            shrinkB=2,
        ),
    )


def training_diagram(out: Path) -> None:
    fig, ax = plt.subplots(figsize=(12.2, 6.4), dpi=170)
    fig.patch.set_facecolor(SURF)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(
        0.005,
        0.965,
        "How the constraint function is trained",
        fontsize=15,
        color=INK,
        weight="bold",
    )
    ax.text(
        0.005,
        0.918,
        "Inverse constraint inference: the constraint is whatever explains why the expert acted differently from the current policy.",
        fontsize=10,
        color=MUTED,
    )

    # the two data sources
    box(
        ax,
        0.02,
        0.70,
        0.235,
        0.13,
        "Expert trajectories\nagent runs judged clean",
        "#e8f3ee",
        SAFE,
    )
    box(
        ax,
        0.02,
        0.44,
        0.235,
        0.13,
        "Policy trajectories\nfresh rollouts of the\ncurrent policy",
        "#fdeceb",
        VIOLATION,
    )

    # encoder
    box(
        ax,
        0.325,
        0.565,
        0.175,
        0.20,
        "Frozen encoder\n\nQwen2.5-1.5B\none vector per step\n(weights never updated)",
        "#f4f6f8",
        "#B9C1CB",
        fontsize=9,
    )

    # feasibility function
    box(
        ax,
        0.565,
        0.565,
        0.185,
        0.20,
        "Feasibility function\nφ(state, action)\n\nprobability this action\nwas safe to take",
        "#e7effa",
        ACCENT,
        fontsize=9.5,
    )

    # the update
    box(
        ax,
        0.80,
        0.70,
        0.185,
        0.115,
        "push φ  up\non expert steps",
        "#e8f3ee",
        SAFE,
        fontsize=10,
    )
    box(
        ax,
        0.80,
        0.545,
        0.185,
        0.115,
        "push φ  down\non policy steps",
        "#fdeceb",
        VIOLATION,
        fontsize=10,
    )

    arrow(ax, (0.255, 0.765), (0.325, 0.70), color=SAFE)
    arrow(ax, (0.255, 0.505), (0.325, 0.62), color=VIOLATION)
    arrow(ax, (0.50, 0.665), (0.565, 0.665), color=MUTED)
    arrow(ax, (0.75, 0.70), (0.80, 0.757), color=SAFE)
    arrow(ax, (0.75, 0.63), (0.80, 0.602), color=VIOLATION)

    ax.text(
        0.893,
        0.855,
        "the two-player gap:\nonly the difference is learned",
        fontsize=8.8,
        color=MUTED,
        ha="center",
        style="italic",
    )

    # cost
    box(
        ax,
        0.565,
        0.285,
        0.185,
        0.13,
        "Cost of a trajectory\nc(τ) = − Σ log φ\n\nadds up over steps",
        "#f4f6f8",
        "#B9C1CB",
        fontsize=9.5,
    )
    arrow(ax, (0.6575, 0.565), (0.6575, 0.415), color=MUTED)

    # policy training
    box(
        ax,
        0.29,
        0.10,
        0.28,
        0.135,
        "Fine-tune the policy\nmaximise task reward\nsubject to  E[c(τ)] ≤ ε",
        "#e7effa",
        ACCENT,
        fontsize=9.5,
    )
    arrow(ax, (0.565, 0.35), (0.57, 0.22), color=MUTED, rad=-0.15)

    # the loop back
    arrow(ax, (0.29, 0.168), (0.1375, 0.44), color=VIOLATION, rad=0.28, lw=1.8, ls="--")
    ax.text(
        0.105,
        0.285,
        "the loop that makes it ICRL:\nthe policy's new rollouts become\nthe next round's negatives",
        fontsize=9,
        color=VIOLATION,
        ha="center",
        style="italic",
    )

    ax.text(
        0.005,
        0.025,
        "A single fit on a fixed set of safe and unsafe traces is the first pass of this loop only. Closing the loop is what keeps the constraint valid as the policy moves away from the data it was fitted on.",
        fontsize=8.8,
        color=MUTED,
    )

    fig.tight_layout()
    fig.savefig(out, facecolor=SURF)
    print("saved", out)


def behaviour_diagram(transfer: Path, out: Path, threshold: float = 0.5) -> None:
    report = json.loads(transfer.read_text())
    rows = [r for arm in report["arms"].values() for r in arm["rows"]]
    rng = random.Random(0)
    safe = [r["C"] for r in rows if not r["label"]]
    bad = [r["C"] for r in rows if r["label"]]

    fig, ax = plt.subplots(figsize=(12.2, 5.4), dpi=170)
    fig.patch.set_facecolor(SURF)
    ax.set_facecolor(SURF)

    fig.text(0.02, 0.945, "Can it tell a safe trace from an unsafe one?",
             fontsize=15, color=INK, weight="bold")
    fig.text(
        0.02, 0.855,
        f"Every one of the {len(rows)} evaluated traces, placed by the score the constraint function gave it.\n"
        "Colour is what the two judges said, which the model never saw.",
        fontsize=10, color=MUTED, linespacing=1.6,
    )

    for values, y, color, label in (
        (safe, 0.68, SAFE, "judged safe"),
        (bad, 0.26, VIOLATION, "judged to contain a violation"),
    ):
        ys = [y + rng.uniform(-0.13, 0.13) for _ in values]
        ax.scatter(
            values,
            ys,
            s=26,
            color=color,
            alpha=0.5,
            edgecolors="none",
            zorder=3,
            label=label,
        )
        ax.text(
            -0.055,
            y,
            f"{label}\nn = {len(values)}",
            ha="right",
            va="center",
            fontsize=10,
            color=color,
            linespacing=1.5,
        )

    ax.axvline(threshold, color=INK, lw=1.5, ls="--", zorder=4)
    ax.text(
        threshold,
        1.0,
        "  the line the training loop draws",
        fontsize=9.5,
        color=INK,
        va="center",
    )

    n_clean = sum(1 for r in rows if r["C"] < threshold)
    missed = sum(1 for r in rows if r["C"] < threshold and r["label"])
    caught = len(bad) - missed
    flagged = len(rows) - n_clean

    ax.annotate(
        f"called safe: {n_clean} traces\nonly {missed} of them ({100 * missed / n_clean:.0f}%) hide a violation",
        xy=(0.06, -0.06),
        fontsize=10,
        color=INK,
        ha="left",
        va="top",
    )
    ax.annotate(
        f"called unsafe: {flagged} traces\ncatches {caught} of the {len(bad)} real violations ({100 * caught / len(bad):.0f}%)",
        xy=(0.94, -0.06),
        fontsize=10,
        color=INK,
        ha="right",
        va="top",
    )

    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.28, 1.02)
    ax.set_xlabel(
        "constraint score:  0 means the model saw nothing wrong,  1 means it is sure there was a violation",
        color=MUTED,
        fontsize=10.5,
        labelpad=10,
    )
    ax.set_yticks([])
    ax.xaxis.grid(True, color=GRID, lw=0.9, zorder=0)
    ax.set_axisbelow(True)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=MUTED, length=0)

    fig.subplots_adjust(left=0.205, right=0.985, top=0.73, bottom=0.26)
    fig.text(
        0.02,
        0.05,
        "The traces come from four different policies, none of which produced the data the model was trained on.\n"
        "The gap between the two colours is what makes it useful; the red points on the left are what it misses.",
        fontsize=8.8,
        color=MUTED,
        linespacing=1.6,
    )
    fig.savefig(out, facecolor=SURF)
    print("saved", out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--transfer", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, default=Path("output/plots"))
    ap.add_argument("--date", default="2026-09-07")
    a = ap.parse_args()
    a.out_dir.mkdir(parents=True, exist_ok=True)
    training_diagram(a.out_dir / f"{a.date}_constraint_training.png")
    behaviour_diagram(a.transfer, a.out_dir / f"{a.date}_constraint_behaviour.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
