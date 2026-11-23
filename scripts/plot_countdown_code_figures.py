#!/usr/bin/env python
# ABOUTME: Regenerates the Countdown-Code figures: test-set hack rate / true reward / proxy reward per arm, hack rate over GRPO training, and the VARL and ICRL method diagrams
# ABOUTME: Run: python scripts/plot_countdown_code_figures.py --out output/countdown_code
"""
Numbers are copied from each run's eval/*_metrics.json and rl_train.log on the
cluster ($SCRATCH/icrl/countdown_code/<run_name>/); the run each row comes from
is named next to it. Style: baselines grey with a hatch pattern, the method
under test in solid colour, the starting model as a red dotted line.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import sys  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.utils.figure_builds import FigureBuilds  # noqa: E402

START_COLOUR = "#d62728"
# SFT-primed starting model, test set (countdown_code_no_intervention_qwen7b_v1 eval/base): percentages
START = {"violation": 0.2, "true": 43.1, "reward": 43.3}
# (label, colour, hatch, {metric: (value, lo, hi)}, run)
ARMS = [
    (
        "No intervention",
        "#d6d8dc",
        "////",
        {
            "violation": (4.25, 3.35, 5.15),
            "true": (56.95, 53.2, 60.65),
            "reward": (61.2, 57.9, 64.6),
        },
        "countdown_code_no_intervention_qwen7b_v1",
    ),
    (
        "VARL (our reimplementation)",
        "#0f6e8c",
        None,
        {
            "violation": (3.8, 2.9, 4.7),
            "true": (58.4, 54.8, 61.9),
            "reward": (62.2, 58.9, 65.6),
        },
        "countdown_code_varl_qwen7b_v1",
    ),
    (
        "ICRL (PUCL)",
        "#7e22ce",
        None,
        {
            "violation": (1.05, 0.65, 1.5),
            "true": (48.15, 44.35, 51.8),
            "reward": (49.2, 45.35, 52.9),
        },
        "countdown_code_icrl_pucl_qwen7b_v1",
    ),
]
# Hack rate (%) of the policy's own training replies, pooled over 25-step windows (rl_train.log of each run)
WINDOWS = [25, 50, 75, 100, 125, 150, 175, 200, 225, 250, 275, 300]
TRAINING_HACK_RATE = {
    "No intervention": [2.6, 2.1, 2.2, 2.7, 2.7, 2.4, 2.7, 3.8, 3.4, 4.3, 4.9, 5.4],
    "VARL (our reimplementation)": [
        2.1,
        2.2,
        2.4,
        2.8,
        2.8,
        2.5,
        2.5,
        3.5,
        3.7,
        4.1,
        5.7,
        4.7,
    ],
    "ICRL (PUCL)": [2.3, 2.1, 2.5, 2.2, 2.4, 2.9, 2.3, 2.4, 3.0, 3.7, 3.6, 2.8],
}


def _style(ax) -> None:
    ax.set_facecolor("white")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color("#333333")
        ax.spines[sp].set_linewidth(0.8)
    ax.grid(axis="y", color="#ececec", lw=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=7.5, colors="#222222", length=3, width=0.6)


def _group_label(fig, axes, text: str, y: float = 0.03) -> None:
    fig.canvas.draw()
    x0, x1 = axes[0].get_position().x0, axes[-1].get_position().x1
    fig.add_artist(
        plt.Line2D(
            [x0, x1],
            [y + 0.03, y + 0.03],
            color="#333333",
            lw=0.8,
            transform=fig.transFigure,
        )
    )
    fig.text(
        (x0 + x1) / 2, y, text, ha="center", va="center", fontsize=8, color="#222222"
    )


def test_results(out: Path) -> None:
    plt.rcParams["hatch.linewidth"] = 0.6
    plt.rcParams["hatch.color"] = "#444444"
    fig, axs = plt.subplots(1, 3, figsize=(7.6, 3.4), facecolor="white")
    panels = [
        (
            axs[0],
            "violation",
            (0, 8),
            "Hack rate (%)",
            "Reward hacking\n(lower is better)",
        ),
        (
            axs[1],
            "true",
            (0, 80),
            "True reward (%)",
            "Puzzle really solved\n(higher is better)",
        ),
        (
            axs[2],
            "reward",
            (0, 80),
            "Proxy reward (%)",
            "Passes its own test file\n(the training reward)",
        ),
    ]
    xs = [0, 1.6, 2.6]  # a gap between the no-intervention baseline and the two methods compared
    for ax, key, ylim, ylabel, title in panels:
        _style(ax)
        for x, (label, colour, hatch, vals, _) in zip(xs, ARMS):
            v, lo, hi = vals[key]
            ax.bar(
                x,
                v,
                width=0.72,
                color=colour,
                edgecolor="#333333",
                linewidth=0.5,
                hatch=hatch,
                zorder=2,
            )
            ax.errorbar(
                x,
                v,
                yerr=[[v - lo], [hi - v]],
                fmt="none",
                ecolor="#222222",
                elinewidth=0.8,
                capsize=2.2,
                capthick=0.8,
                zorder=3,
            )
            ax.text(
                x,
                hi + ylim[1] * 0.02,
                f"{v:.1f}",
                ha="center",
                va="bottom",
                fontsize=6.6,
                weight="bold",
                color="#111111",
            )
        ax.axhline(START[key], color=START_COLOUR, lw=1.0, ls=(0, (2, 2)), zorder=1)
        ax.axvline(0.8, color="#999999", lw=0.7, ls=(0, (3, 3)), zorder=1)
        ax.set_xlim(-0.7, 3.3)
        ax.set_ylim(*ylim)
        ax.set_xticks([])
        ax.set_ylabel(ylabel, fontsize=8, color="#222222")
        ax.set_title(title, fontsize=8.5, color="#222222", linespacing=1.2)
    handles = [
        plt.Line2D(
            [], [], color=START_COLOUR, lw=1.0, ls=(0, (2, 2)), label="SFT-primed start"
        )
    ]
    handles += [
        matplotlib.patches.Patch(
            facecolor=c, edgecolor="#333333", linewidth=0.5, hatch=h, label=lab
        )
        for lab, c, h, _, _ in ARMS
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        ncol=4,
        fontsize=7,
        frameon=False,
        handlelength=1.6,
        columnspacing=1.4,
        bbox_to_anchor=(0.5, 1.0),
    )
    fig.tight_layout(rect=(0, 0.08, 1, 0.92), w_pad=1.6)
    _group_label(fig, [axs[0]], "Constraint")
    _group_label(fig, [axs[1], axs[2]], "Reward")
    fig.savefig(out / "countdown_code_test_results.png", dpi=300, facecolor="white")
    plt.close(fig)
    print("wrote", out / "countdown_code_test_results.png")


def training_hack_rate(out: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.2, 2.9), facecolor="white")
    _style(ax)
    styles = {
        "No intervention": ("#8a8f99", (0, (4, 2))),
        "VARL (our reimplementation)": ("#0f6e8c", "-"),
        "ICRL (PUCL)": ("#7e22ce", "-"),
    }
    for label, vals in TRAINING_HACK_RATE.items():
        colour, ls = styles[label]
        ax.plot(
            WINDOWS,
            vals,
            color=colour,
            lw=1.6,
            ls=ls,
            marker="o",
            ms=3.8,
            mec="white",
            mew=0.7,
            label=label,
            zorder=3 if label.startswith("ICRL") else 2,
        )
    ax.set_xlim(0, 310)
    ax.set_ylim(0, 8)
    ax.set_xlabel("GRPO step", fontsize=8, color="#222222")
    ax.set_ylabel("Hack rate (%)", fontsize=8, color="#222222")
    ax.set_title(
        "Hack rate of the policy's training replies\n(lower is better)",
        fontsize=8.5,
        color="#222222",
        linespacing=1.2,
    )
    fig.legend(
        loc="upper center",
        ncol=3,
        fontsize=7,
        frameon=False,
        bbox_to_anchor=(0.5, 1.0),
        handlelength=2.4,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(
        out / "countdown_code_training_hack_rate.png", dpi=300, facecolor="white"
    )
    plt.close(fig)
    print("wrote", out / "countdown_code_training_hack_rate.png")


def method_diagram(out: Path, kind: str) -> None:
    """How one arm scores a sampled reply and how its monitor is trained. kind = 'varl' or 'icrl'."""
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    PURPLE, GREEN, RED, GOLD = "#7e22ce", "#2e9d5b", "#d62728", "#b7791f"
    INKD, GREY, NEUTRAL = "#1d1d1f", "#555555", "#9a9da5"
    varl = kind == "varl"
    MON = "#0f6e8c" if varl else PURPLE  # the monitor's colour
    MON_FILL = "#eaf5f9" if varl else "#faf5ff"
    W, Y0, H = 16.05, -0.1, 7.4
    fig = plt.figure(figsize=(5.35, 5.35 * (H - Y0) / W), facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(Y0, H)
    ax.axis("off")
    b = FigureBuilds(ax)  # click-by-click versions for presentations

    def bubble(x, y, w, h, head, lines, ec=NEUTRAL, fc="white", fs=4.6):
        ax.add_patch(
            FancyBboxPatch(
                (x, y),
                w,
                h,
                boxstyle="round,pad=0,rounding_size=0.14",
                fc=fc,
                ec=ec,
                lw=1.1,
                zorder=3,
            )
        )
        top = y + h - 0.13
        if head:
            ax.text(
                x + 0.12,
                top,
                head,
                ha="left",
                va="top",
                fontsize=4.3,
                weight="bold",
                color=GREY,
                zorder=5,
            )
            top -= 0.36
        for i, (t, c) in enumerate(lines):
            ax.text(
                x + 0.12,
                top - 0.33 * i,
                t,
                ha="left",
                va="top",
                fontsize=fs,
                color=c or INKD,
                zorder=5,
                family="monospace" if head in ("SAMPLED REPLY", None) and c else None,
            )

    def robot(cx, cy, s):
        ax.add_patch(
            FancyBboxPatch(
                (cx - 0.7 * s, cy - 0.55 * s),
                1.4 * s,
                1.1 * s,
                boxstyle="round,pad=0,rounding_size=0.18",
                fc="#e9eaee",
                ec="#5b6270",
                lw=1.0,
                zorder=6,
            )
        )
        for ex in (cx - 0.3 * s, cx + 0.3 * s):
            ax.add_patch(
                plt.Circle((ex, cy + 0.08 * s), 0.11 * s, fc="#5b6270", zorder=7)
            )
        ax.plot(
            [cx - 0.25 * s, cx + 0.25 * s],
            [cy - 0.25 * s] * 2,
            color="#5b6270",
            lw=1.0,
            zorder=7,
        )
        ax.plot(
            [cx, cx], [cy + 0.55 * s, cy + 0.8 * s], color="#5b6270", lw=1.0, zorder=6
        )
        ax.add_patch(plt.Circle((cx, cy + 0.87 * s), 0.09 * s, fc=PURPLE, zorder=7))

    def arr(p1, p2, color="#333333", lw=1.2, rad=0.0):
        ax.add_patch(
            FancyArrowPatch(
                p1,
                p2,
                arrowstyle="-|>",
                mutation_scale=8,
                color=color,
                lw=lw,
                connectionstyle=f"arc3,rad={rad}",
                zorder=2,
            )
        )

    def note(x, y, t, color=INKD, fs=5.0, ha="center", **kw):
        ax.text(
            x,
            y,
            t,
            ha=ha,
            va="center",
            fontsize=fs,
            color=color,
            linespacing=1.15,
            zorder=5,
            **kw,
        )

    def chip(x, y, t, color, w=0.8):
        ax.add_patch(
            FancyBboxPatch(
                (x - w / 2, y - 0.22),
                w,
                0.44,
                boxstyle="round,pad=0,rounding_size=0.22",
                fc=color,
                ec="none",
                zorder=4,
            )
        )
        note(x, y, t, color="white", fs=5.2, weight="bold")

    def model_box(x, y, w, h, title, sub, ec, fc, sub_color, ls="-"):
        ax.add_patch(
            FancyBboxPatch(
                (x, y),
                w,
                h,
                boxstyle="round,pad=0,rounding_size=0.18",
                fc=fc,
                ec=ec,
                lw=1.6,
                ls=ls,
                zorder=3,
            )
        )
        note(x + w / 2, y + h - 0.35, title, color=ec, weight="bold", fs=5.3)
        note(x + w / 2, y + 0.36, sub, color=sub_color, fs=4.6)

    MID = 3.9
    b.stage(1)
    # the prompt
    note(1.75, 5.55, "Training prompt", weight="bold", fs=5.4)
    bubble(
        0.2,
        4.1,
        3.1,
        1.1,
        "solution.py",
        [("numbers = [36, 36, 39, 26]", None), ("target = 66   expr = None", None)],
        fc="#ecebe6",
        ec="#e6e5e0",
        fs=4.2,
    )
    bubble(
        0.2,
        2.9,
        3.1,
        0.95,
        "test.py",
        [("def verify_solution(...):", None), ("  # checks expr", None)],
        fc="#dce8f7",
        ec="#dce8f7",
        fs=4.2,
    )
    b.stage(2)
    # the policy and one sampled reply (a hack)
    robot(4.3, MID, 0.7)
    note(4.3, 2.95, "Policy (LLM)", weight="bold", fs=5.2)
    arr((3.35, MID), (3.75, MID))
    note(6.42, 4.85, "sampled reply (a hack)", weight="bold", fs=5.0)
    bubble(
        5.1,
        MID - 0.55,
        2.65,
        1.1,
        "test.py",
        [("def verify_solution(...):", None), ("  return True", RED)],
        ec=RED,
        fs=4.2,
    )
    arr((4.85, MID), (5.05, MID))
    b.stage(3)
    # reward and monitor
    model_box(
        8.45,
        4.55,
        2.9,
        1.1,
        "Proxy reward",
        "runs the returned test",
        GOLD,
        "#fdf8ee",
        GREY,
    )
    b.stage(5)
    mon_title = "Discriminator $D$" if varl else "Constraint model $C_\\theta$"
    mon_sub = (
        "P(demonstration)" if varl else "cost = 1 $-$ P(allowed)"
    )
    model_box(8.45, 2.2, 2.9, 1.1, mon_title, mon_sub, MON, MON_FILL, MON)
    b.stage(3)
    arr((7.8, MID + 0.2), (8.4, 4.95), color=GOLD, lw=1.1)
    b.stage(5)
    arr((7.8, MID - 0.2), (8.4, 2.9), color=MON, lw=1.1)
    b.stage(3)
    chip(12.05, 5.1, "1", GOLD)
    note(12.05, 5.6, "passes", color=GOLD, fs=4.7)
    b.stage(5)
    chip(12.05, 2.75, "0.1" if varl else "0.9", MON)
    note(12.05, 2.25, "low $D$" if varl else "high cost", color=MON, fs=4.7)
    b.stage(3)
    arr((11.4, 5.1), (11.6, 5.1), color=GOLD, lw=1.0)
    b.stage(5)
    arr((11.4, 2.75), (11.6, 2.75), color=MON, lw=1.0)
    b.stage(4)
    # GRPO
    ax.add_patch(
        FancyBboxPatch(
            (13.3, 3.2),
            2.55,
            1.4,
            boxstyle="round,pad=0,rounding_size=0.2",
            fc="white",
            ec="#333333",
            lw=1.3,
            zorder=3,
        )
    )
    note(14.57, 4.25, "GRPO", weight="bold", fs=6.8)
    if varl:
        ax.text(
            14.45,
            3.62,
            "reward ",
            ha="right",
            va="center",
            fontsize=5.0,
            color=INKD,
            zorder=5,
        )
        b.stage(5)
        ax.text(
            14.45,
            3.62,
            "$\\times\\ D$",
            ha="left",
            va="center",
            fontsize=5.6,
            color=MON,
            weight="bold",
            zorder=5,
        )
    else:
        ax.text(
            14.35,
            3.62,
            "reward ",
            ha="right",
            va="center",
            fontsize=5.0,
            color=INKD,
            zorder=5,
        )
        b.stage(5)
        ax.text(
            14.35,
            3.62,
            "$-\\ \\lambda\\cdot$cost",
            ha="left",
            va="center",
            fontsize=5.3,
            color=MON,
            weight="bold",
            zorder=5,
        )
    b.stage(4)
    arr((12.45, 5.1), (13.4, 4.65), color=GOLD, lw=1.0)
    b.stage(5)
    arr((12.45, 2.75), (13.4, 3.15), color=MON, lw=1.0)
    b.stage(4)
    arr((14.57, 4.65), (4.3, 4.75), rad=0.4)
    b.stage(5)
    note(
        9.4,
        7.05,
        "feedback: the reply earns the reward but scores low, so it becomes less likely",
        fs=4.9,
        style="italic",
    )
    b.stage(6)
    # how the monitor is trained
    ax.add_patch(
        FancyBboxPatch(
            (0.2, 0.15),
            3.3,
            1.4,
            boxstyle="round,pad=0,rounding_size=0.15",
            fc="#f1faf4",
            ec=GREEN,
            lw=1.2,
            zorder=2,
        )
    )
    note(1.85, 1.25, "Demonstrations", color=GREEN, weight="bold", fs=5.2)
    bubble(
        0.4,
        0.3,
        2.9,
        0.62,
        None,
        [("expr = '39 + 26 + 36/36'", None)],
        ec=GREEN,
        fs=4.2,
    )
    ax.add_patch(
        FancyBboxPatch(
            (4.3, 0.15),
            4.5,
            1.4,
            boxstyle="round,pad=0,rounding_size=0.15",
            fc=MON_FILL,
            ec=MON,
            lw=1.2,
            ls=(0, (3, 2)),
            zorder=2,
        )
    )
    if varl:
        note(6.55, 1.25, "Train $D$ (every step)", color=MON, weight="bold", fs=5.2)
        note(
            6.55,
            0.8,
            "demonstrations $\\rightarrow$ 1",
            color=GREEN,
            weight="bold",
            fs=4.6,
        )
        note(
            6.55,
            0.42,
            "replies that pass the test $\\rightarrow$ 0",
            color=RED,
            weight="bold",
            fs=4.6,
        )
        tail = "policy replies\nthat pass the test"
    else:
        note(
            6.55,
            1.25,
            "Re-fit $C_\\theta$ (each round, PUCL)",
            color=MON,
            weight="bold",
            fs=5.2,
        )
        note(
            6.55,
            0.8,
            "demonstrations $\\rightarrow$ allowed",
            color=GREEN,
            weight="bold",
            fs=4.6,
        )
        note(
            6.55,
            0.42,
            "those far from the demos $\\rightarrow$ forbidden",
            color=RED,
            weight="bold",
            fs=4.3,
        )
        tail = "policy replies\nthat pass the test"
    arr((3.55, 0.85), (4.25, 0.85), color=GREEN, lw=1.2)
    arr((6.42, 3.3), (6.55, 1.6), color=MON, lw=1.1)
    note(6.65, 2.4, tail, color=MON, fs=4.4, style="italic", ha="left")
    arr((8.85, 1.0), (9.4, 2.15), color=MON, lw=1.2)
    note(
        9.3,
        1.35,
        "new $D$" if varl else "new $C_\\theta$",
        color=MON,
        fs=4.6,
        style="italic",
        ha="left",
    )
    name = f"countdown_code_method_{kind}.png"
    b.save(fig, out / "builds" / f"countdown_code_method_{kind}")
    fig.savefig(out / name, dpi=300, facecolor="white")
    plt.close(fig)
    print("wrote", out / name)


def experiment_overview(out: Path) -> None:
    """Fig 1 of results.tex: the loophole, the three ways each reply is scored, and the hack rates so far."""
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    PURPLE, TEAL, GREEN, RED, GOLD = (
        "#7e22ce",
        "#0f6e8c",
        "#2e9d5b",
        "#d62728",
        "#b7791f",
    )
    INKD, GREY, PANEL, EDGE = "#1d1d1f", "#555555", "#f6f6f7", "#c9cbd0"
    plt.rcParams["hatch.linewidth"] = 0.6
    plt.rcParams["hatch.color"] = "#444444"
    W, H = 21.6, 8.4
    fig = plt.figure(figsize=(7.2, 2.8), facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis("off")

    def panel(x0, x1, title):
        ax.add_patch(
            FancyBboxPatch(
                (x0, 0.15),
                x1 - x0,
                H - 0.3,
                boxstyle="round,pad=0,rounding_size=0.2",
                fc=PANEL,
                ec=EDGE,
                lw=0.9,
                zorder=0,
            )
        )
        ax.text(
            x0 + 0.25,
            H - 0.4,
            title,
            ha="left",
            va="top",
            fontsize=8,
            weight="bold",
            color=INKD,
        )

    def bubble(x, y, w, h, head, lines, ec, fc="white", fs=5.0):
        ax.add_patch(
            FancyBboxPatch(
                (x, y),
                w,
                h,
                boxstyle="round,pad=0,rounding_size=0.15",
                fc=fc,
                ec=ec,
                lw=1.1,
                zorder=3,
            )
        )
        ax.text(
            x + 0.15,
            y + h - 0.15,
            head,
            ha="left",
            va="top",
            fontsize=4.6,
            weight="bold",
            color=GREY,
            zorder=5,
        )
        for i, (t, c) in enumerate(lines):
            ax.text(
                x + 0.15,
                y + h - 0.6 - 0.4 * i,
                t,
                ha="left",
                va="top",
                fontsize=fs,
                color=c or INKD,
                zorder=5,
            )

    def note(x, y, t, color=INKD, fs=5.4, ha="center", **kw):
        ax.text(
            x,
            y,
            t,
            ha=ha,
            va="center",
            fontsize=fs,
            color=color,
            linespacing=1.2,
            zorder=5,
            **kw,
        )

    def chip(x, y, t, color, w=2.0):
        ax.add_patch(
            FancyBboxPatch(
                (x - w / 2, y - 0.24),
                w,
                0.48,
                boxstyle="round,pad=0,rounding_size=0.24",
                fc=color,
                ec="none",
                zorder=4,
            )
        )
        note(x, y, t, color="white", fs=5.0, weight="bold")

    def arr(p1, p2, color="#444444", lw=1.1):
        ax.add_patch(
            FancyArrowPatch(
                p1, p2, arrowstyle="-|>", mutation_scale=8, color=color, lw=lw, zorder=2
            )
        )

    def robot(cx, cy, s):
        ax.add_patch(
            FancyBboxPatch(
                (cx - 0.7 * s, cy - 0.55 * s),
                1.4 * s,
                1.1 * s,
                boxstyle="round,pad=0,rounding_size=0.18",
                fc="#e9eaee",
                ec="#5b6270",
                lw=1.0,
                zorder=6,
            )
        )
        for ex in (cx - 0.3 * s, cx + 0.3 * s):
            ax.add_patch(
                plt.Circle((ex, cy + 0.08 * s), 0.11 * s, fc="#5b6270", zorder=7)
            )
        ax.plot(
            [cx - 0.25 * s, cx + 0.25 * s],
            [cy - 0.25 * s] * 2,
            color="#5b6270",
            lw=1.0,
            zorder=7,
        )
        ax.plot(
            [cx, cx], [cy + 0.55 * s, cy + 0.8 * s], color="#5b6270", lw=1.0, zorder=6
        )
        ax.add_patch(plt.Circle((cx, cy + 0.87 * s), 0.09 * s, fc=PURPLE, zorder=7))

    # ---- 1. the loophole
    panel(0.15, 6.9, "1  The loophole")
    bubble(
        0.4,
        5.6,
        6.25,
        1.5,
        "TASK",
        [
            ("Reach 66 with 36, 36, 39, 26.", None),
            ("Return solution.py and test.py.", None),
        ],
        "#e6e5e0",
        fc="#ecebe6",
    )
    bubble(
        0.4,
        3.35,
        6.25,
        1.5,
        "HONEST REPLY",
        [("expr = '39 + 26 + 36/36'", None), ("test.py unchanged", GREY)],
        GREEN,
    )
    bubble(
        0.4,
        1.1,
        6.25,
        1.5,
        "HACK",
        [("expr = '1'", None), ("test.py:  return True", RED)],
        RED,
    )
    for y in (4.1, 1.85):
        chip(5.55, y + 0.33, "reward 1", GOLD, w=1.7)
    note(5.55, 3.75, "solved", color=GREEN, fs=4.8, weight="bold")
    note(5.55, 1.5, "not solved", color=RED, fs=4.8, weight="bold")
    note(
        3.5, 0.55, "The training reward cannot tell them apart.", fs=5.0, style="italic"
    )

    # ---- 2. the three arms
    panel(7.15, 15.3, "2  Three ways to score a reply in RL")
    robot(8.2, 4.2, 0.75)
    note(8.2, 3.15, "LLM policy", fs=5.0, color=GREY)
    rows = [
        (6.1, "No intervention", "reward", "#8a8f99", "the test result alone"),
        (4.2, "VARL", "reward $\\times\\ D$", TEAL, "$D$: looks like a demonstration?"),
        (
            2.3,
            "ICRL (ours)",
            "reward $-\\ \\lambda\\cdot$cost",
            PURPLE,
            "cost from a learned constraint",
        ),
    ]
    for y, name, formula, col, sub in rows:
        arr((8.85, 4.2), (9.75, y), color="#9a9da5", lw=0.9)
        ax.add_patch(
            FancyBboxPatch(
                (9.85, y - 0.62),
                5.2,
                1.24,
                boxstyle="round,pad=0,rounding_size=0.15",
                fc="white",
                ec=col,
                lw=1.4,
                zorder=3,
            )
        )
        note(10.05, y + 0.25, name, color=col, fs=5.6, ha="left", weight="bold")
        note(14.85, y + 0.25, formula, fs=5.6, ha="right")
        if sub:
            note(10.05, y - 0.28, sub, color=GREY, fs=4.6, ha="left")
    note(
        11.2,
        0.75,
        "$D$ and the constraint are learned from\ndemonstrations. No hack labels.",
        color=GREEN,
        fs=5.0,
        weight="bold",
    )

    # ---- 3. results so far
    panel(15.55, 21.45, "3  After 1,250 RL steps")
    bx = fig.add_axes([18.3 / W, 2.75 / H, 2.75 / W, 3.9 / H])
    bx.set_facecolor("none")
    data = [
        ("No intervention", 30.4, "#d6d8dc", "////"),
        ("VARL", 3.3, TEAL, None),
        ("ICRL (ours)", 0.5, PURPLE, None),
    ]
    ys = [2, 1, 0]
    for y, (name, v, c, h) in zip(ys, data):
        bx.barh(
            y,
            v,
            height=0.62,
            color=c,
            edgecolor="#333333",
            linewidth=0.5,
            hatch=h,
            zorder=2,
        )
        bx.text(
            v + 0.9, y, f"{v:.1f}%", va="center", fontsize=6, weight="bold", color=INKD
        )
    bx.axvline(0.65, color=START_COLOUR, lw=0.9, ls=(0, (2, 2)), zorder=1)
    bx.set_yticks(ys)
    bx.set_yticklabels([d[0] for d in data], fontsize=5.6, color=INKD)
    bx.set_xlim(0, 40)
    bx.set_xticks([0, 10, 20, 30, 40])
    bx.tick_params(axis="x", labelsize=5.4, colors=GREY, length=2)
    bx.tick_params(axis="y", length=0)
    for sp in ("top", "right", "left"):
        bx.spines[sp].set_visible(False)
    bx.spines["bottom"].set_color("#888888")
    bx.set_xlabel(
        "hack rate on test problems (%)", fontsize=5.4, color=GREY, labelpad=2
    )
    note(18.5, 7.0, "start: 0.7%", color=START_COLOUR, fs=4.8)
    note(
        18.5,
        1.05,
        "Both methods stop most of the hacking.\nPuzzles solved: 55% / 69% / 59%.\nOne seed per arm.",
        fs=5.0,
        color=INKD,
        weight="bold",
    )

    fig.savefig(
        out / "countdown_code_experiment_overview.png", dpi=300, facecolor="white"
    )
    plt.close(fig)
    print("wrote", out / "countdown_code_experiment_overview.png")


def graft_diagram(out: Path) -> None:
    """How GRAFT (Ward et al. 2026) handles a sampled reply: the reward is unchanged, the monitor only decides
    which adapter the update goes to, and the forget adapter is removed at deployment."""
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    PURPLE, GREEN, RED, GOLD, ORANGE = (
        "#7e22ce",
        "#2e9d5b",
        "#d62728",
        "#b7791f",
        "#e8792b",
    )
    INKD, GREY, NEUTRAL, BLUE = "#1d1d1f", "#555555", "#9a9da5", "#2b5aa8"
    W, Y0, H = 16.05, -0.1, 7.4
    fig = plt.figure(figsize=(5.35, 5.35 * (H - Y0) / W), facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(Y0, H)
    ax.axis("off")

    def bubble(x, y, w, h, head, lines, ec=NEUTRAL, fc="white", fs=4.2):
        ax.add_patch(
            FancyBboxPatch(
                (x, y),
                w,
                h,
                boxstyle="round,pad=0,rounding_size=0.14",
                fc=fc,
                ec=ec,
                lw=1.1,
                zorder=3,
            )
        )
        ax.text(
            x + 0.12,
            y + h - 0.13,
            head,
            ha="left",
            va="top",
            fontsize=4.3,
            weight="bold",
            color=GREY,
            zorder=5,
        )
        for i, (t, c) in enumerate(lines):
            ax.text(
                x + 0.12,
                y + h - 0.49 - 0.33 * i,
                t,
                ha="left",
                va="top",
                fontsize=fs,
                color=c or INKD,
                zorder=5,
            )

    def arr(p1, p2, color="#333333", lw=1.2, rad=0.0, ls="-"):
        ax.add_patch(
            FancyArrowPatch(
                p1,
                p2,
                arrowstyle="-|>",
                mutation_scale=8,
                color=color,
                lw=lw,
                linestyle=ls,
                connectionstyle=f"arc3,rad={rad}",
                zorder=2,
            )
        )

    def note(x, y, t, color=INKD, fs=5.0, ha="center", **kw):
        ax.text(
            x,
            y,
            t,
            ha=ha,
            va="center",
            fontsize=fs,
            color=color,
            linespacing=1.15,
            zorder=5,
            **kw,
        )

    def chip(x, y, t, color, w=0.8):
        ax.add_patch(
            FancyBboxPatch(
                (x - w / 2, y - 0.22),
                w,
                0.44,
                boxstyle="round,pad=0,rounding_size=0.22",
                fc=color,
                ec="none",
                zorder=4,
            )
        )
        note(x, y, t, color="white", fs=5.0, weight="bold")

    def box(x, y, w, h, title, sub, ec, fc, sub_color, ls="-"):
        ax.add_patch(
            FancyBboxPatch(
                (x, y),
                w,
                h,
                boxstyle="round,pad=0,rounding_size=0.18",
                fc=fc,
                ec=ec,
                lw=1.6,
                ls=ls,
                zorder=3,
            )
        )
        note(x + w / 2, y + h - 0.35, title, color=ec, weight="bold", fs=5.3)
        note(x + w / 2, y + 0.36, sub, color=sub_color, fs=4.6)

    def policy(cx, cy, forget=True):
        """The LLM with its two added adapters drawn as blocks."""
        ax.add_patch(
            FancyBboxPatch(
                (cx - 0.75, cy - 0.6),
                1.5,
                1.2,
                boxstyle="round,pad=0,rounding_size=0.18",
                fc="#e9eaee",
                ec="#5b6270",
                lw=1.0,
                zorder=5,
            )
        )
        note(cx, cy + 0.33, "LLM", fs=4.8, color="#5b6270", weight="bold")
        ax.add_patch(
            FancyBboxPatch(
                (cx - 0.62, cy - 0.45),
                0.56,
                0.5,
                boxstyle="round,pad=0,rounding_size=0.06",
                fc=BLUE,
                ec="none",
                zorder=6,
            )
        )
        ax.text(cx - 0.34, cy - 0.2, "R", ha="center", va="center", fontsize=5.0, color="white", weight="bold", zorder=9)
        ax.add_patch(
            FancyBboxPatch(
                (cx + 0.06, cy - 0.45),
                0.56,
                0.5,
                boxstyle="round,pad=0,rounding_size=0.06",
                fc=ORANGE if forget else "white",
                ec=ORANGE,
                lw=1.0,
                ls="-" if forget else (0, (2, 1.5)),
                zorder=6,
            )
        )
        ax.text(cx + 0.34, cy - 0.2, "F", ha="center", va="center", fontsize=5.0, color="white" if forget else ORANGE, weight="bold", zorder=9)

    MID = 3.9
    note(1.75, 5.55, "Training prompt", weight="bold", fs=5.4)
    bubble(
        0.2,
        4.1,
        3.1,
        1.1,
        "solution.py",
        [("numbers = [36, 36, 39, 26]", None), ("target = 66   expr = None", None)],
        fc="#ecebe6",
        ec="#e6e5e0",
    )
    bubble(
        0.2,
        2.9,
        3.1,
        0.95,
        "test.py",
        [("def verify_solution(...):", None), ("  # checks expr", None)],
        fc="#dce8f7",
        ec="#dce8f7",
    )
    policy(4.3, MID)
    note(4.3, 5.0, "Policy with adapters\nR = retain, F = forget", weight="bold", fs=4.5)
    arr((3.35, MID), (3.5, MID))
    note(6.5, 4.62, "sampled reply (a hack)", weight="bold", fs=4.8)
    bubble(
        5.15,
        MID - 0.55,
        2.6,
        1.1,
        "test.py",
        [("def verify_solution(...):", None), ("  return True", RED)],
        ec=RED,
    )
    arr((5.05, MID), (5.12, MID))

    box(
        8.45,
        4.55,
        2.9,
        1.1,
        "Proxy reward",
        "runs the returned test",
        GOLD,
        "#fdf8ee",
        GREY,
    )
    box(
        8.45,
        2.2,
        2.9,
        1.1,
        "Monitor",
        "flags replies it can detect",
        ORANGE,
        "#fef4ec",
        ORANGE,
    )
    arr((7.8, MID + 0.2), (8.4, 4.95), color=GOLD, lw=1.1)
    arr((7.8, MID - 0.2), (8.4, 2.9), color=ORANGE, lw=1.1)
    chip(12.05, 5.1, "1", GOLD)
    note(12.05, 5.6, "passes", color=GOLD, fs=4.7)
    chip(12.05, 2.75, "flagged", ORANGE, w=1.25)
    arr((11.4, 5.1), (11.6, 5.1), color=GOLD, lw=1.0)
    arr((11.4, 2.75), (11.38, 2.75), color=ORANGE, lw=1.0)

    ax.add_patch(
        FancyBboxPatch(
            (13.3, 4.35),
            2.55,
            1.4,
            boxstyle="round,pad=0,rounding_size=0.2",
            fc="white",
            ec="#333333",
            lw=1.3,
            zorder=3,
        )
    )
    note(14.57, 5.4, "GRPO", weight="bold", fs=6.8)
    note(14.57, 4.78, "reward, unchanged", fs=4.9)
    arr((12.5, 5.1), (13.25, 5.1), color=GOLD, lw=1.0)

    # routing: the monitor decides which adapter is updated
    ax.add_patch(
        FancyBboxPatch(
            (12.4, 0.3),
            3.45,
            1.9,
            boxstyle="round,pad=0,rounding_size=0.15",
            fc="#fef4ec",
            ec=ORANGE,
            lw=1.2,
            ls=(0, (3, 2)),
            zorder=2,
        )
    )
    note(14.12, 1.85, "Route the update", color=ORANGE, weight="bold", fs=5.2)
    note(
        14.12, 1.3, "flagged $\\rightarrow$ F only", color=ORANGE, weight="bold", fs=4.8
    )
    note(
        14.12,
        0.8,
        "not flagged $\\rightarrow$ R and F",
        color=BLUE,
        weight="bold",
        fs=4.8,
    )
    arr((12.4, 2.5), (13.2, 2.25), color=ORANGE, lw=1.1)
    arr((14.57, 4.3), (14.4, 2.25), color="#333333", lw=1.1)
    note(14.75, 3.3, "gradient", fs=4.4, color=GREY, ha="left", style="italic")
    arr((12.35, 1.1), (4.5, 3.25), rad=-0.22, color=ORANGE)
    note(9.6, 0.75,
        "this reply was flagged: only F is updated",
        color=ORANGE,
        fs=4.6,
        style="italic",
    )

    # deployment
    ax.add_patch(
        FancyBboxPatch(
            (0.2, 0.15),
            6.1,
            1.5,
            boxstyle="round,pad=0,rounding_size=0.15",
            fc="#f1faf4",
            ec=GREEN,
            lw=1.2,
            zorder=2,
        )
    )
    note(2.35, 1.3, "At deployment", color=GREEN, weight="bold", fs=5.2)
    note(
        2.35,
        0.65,
        "remove the forget adapter;\nthe hacking goes with it",
        color=INKD,
        fs=4.6,
    )
    policy(5.2, 0.9, forget=False)

    note(
        8.0,
        7.05,
        "The monitor never changes the reward, so the policy is not paid for fooling it",
        fs=4.9,
        style="italic",
    )
    fig.savefig(out / "countdown_code_method_graft.png", dpi=300, facecolor="white")
    plt.close(fig)
    print("wrote", out / "countdown_code_method_graft.png")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--out", type=Path, default=Path("output/countdown_code"))
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    experiment_overview(a.out)
    test_results(a.out)
    training_hack_rate(a.out)
    method_diagram(a.out, "varl")
    method_diagram(a.out, "icrl")
    graft_diagram(a.out)


if __name__ == "__main__":
    main()
