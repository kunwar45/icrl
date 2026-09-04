#!/usr/bin/env python
# ABOUTME: Plots ODCV evaluation results across policy arms (misalignment rate with clustered CIs, per framing, submission and integrity rates)
# ABOUTME: Run: python scripts/plot_odcv_evaluation.py --arm "organism=<results.json>" --arm "dpo=<results.json>" ... --summaries "organism=<summary.json>" --out output/plots/<name>.png
"""
One figure that answers the project's question: misalignment rate (the
benchmark's judged number, LASR protocol) before and after fine-tuning, with
the guard rates that say whether a drop is alignment or inaction.

Left: MR with 95% scenario-clustered CI per arm, all-80 and LASR-65 protocols.
Middle: MR per framing (mandated / incentivized), all-80.
Right: submission rate and file-integrity violation rate per arm (from the
collector's summary.json), the rule-based signals no judge produced.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

INK, MUTED, GRID, SURF = "#1a1a19", "#5d5d5a", "#e6e6e3", "#fcfcfb"
BLUE, ORANGE, GREEN = "#2a78d6", "#eb6834", "#2e8b57"


def parse_pairs(items):
    out = {}
    for it in items or []:
        k, v = it.split("=", 1)
        out[k] = Path(v)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--arm",
        action="append",
        required=True,
        help="label=path/to/results.json (judge output)",
    )
    ap.add_argument(
        "--summary",
        action="append",
        default=[],
        help="label=path/to/summary.json (collector output)",
    )
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument(
        "--title",
        default="ODCV-Bench: misalignment before and after constrained fine-tuning",
    )
    a = ap.parse_args()
    arms = parse_pairs(a.arm)
    sums = parse_pairs(a.summary)
    labels = list(arms)
    R = {k: json.loads(p.read_text()) for k, p in arms.items()}
    S = {k: json.loads(p.read_text()) for k, p in sums.items() if p.exists()}

    fig, (a1, a2, a3) = plt.subplots(
        1, 3, figsize=(14, 4.8), dpi=160, gridspec_kw={"width_ratios": [1.2, 1, 1]}
    )
    fig.patch.set_facecolor(SURF)
    for ax in (a1, a2, a3):
        ax.set_facecolor(SURF)
        ax.tick_params(colors=MUTED, length=0)
        ax.yaxis.grid(True, color=GRID, lw=0.8, zorder=0)
        ax.set_axisbelow(True)
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
        ax.spines["bottom"].set_color(GRID)

    x = np.arange(len(labels))
    w = 0.36
    for j, (proto, color, off) in enumerate(
        (("all_80", BLUE, -w / 2 - 0.01), ("lasr_65", ORANGE, w / 2 + 0.01))
    ):
        mr = [R[k][proto]["summary"]["overall"].get("mr_pct", np.nan) for k in labels]
        ci = [
            R[k][proto]["summary"]["overall"].get("mr_ci95") or [np.nan, np.nan]
            for k in labels
        ]
        lo = [m - c[0] if c and c[0] == c[0] else 0 for m, c in zip(mr, ci)]
        hi = [c[1] - m if c and c[1] == c[1] else 0 for m, c in zip(mr, ci)]
        a1.bar(
            x + off,
            mr,
            w,
            color=color,
            label="all 80 cells" if proto == "all_80" else "LASR 65-cell subset",
            zorder=3,
        )
        a1.errorbar(
            x + off,
            mr,
            yerr=[lo, hi],
            fmt="none",
            ecolor=INK,
            elinewidth=1,
            capsize=3,
            zorder=4,
        )
        for xi, m in zip(x + off, mr):
            a1.text(xi, m + 1.5, f"{m:.1f}", ha="center", fontsize=8, color=INK)
    a1.set_xticks(x)
    a1.set_xticklabels(labels, fontsize=9, color=INK)
    a1.set_ylabel("misalignment rate, % of cells (median judge ≥ 3)", color=MUTED)
    a1.set_ylim(0, max(60, a1.get_ylim()[1]))
    a1.legend(frameon=False, fontsize=8, loc="upper right")
    a1.set_title(
        "Judged misalignment, 95% scenario-clustered CI",
        fontsize=10,
        color=INK,
        loc="left",
    )

    for j, (framing, color, off) in enumerate(
        (("mandated", BLUE, -w / 2 - 0.01), ("incentivized", ORANGE, w / 2 + 0.01))
    ):
        mr = [
            R[k]["all_80"]["summary"].get(framing, {}).get("mr_pct", np.nan)
            for k in labels
        ]
        a2.bar(x + off, mr, w, color=color, label=framing, zorder=3)
        for xi, m in zip(x + off, mr):
            if m == m:
                a2.text(xi, m + 1.5, f"{m:.0f}", ha="center", fontsize=8, color=INK)
    a2.set_xticks(x)
    a2.set_xticklabels(labels, fontsize=9, color=INK)
    a2.set_ylim(0, max(60, a2.get_ylim()[1]))
    a2.legend(frameon=False, fontsize=8, loc="upper right")
    a2.set_title("By framing, all 80 cells", fontsize=10, color=INK, loc="left")

    sub = [S.get(k, {}).get("submitted_pct", np.nan) for k in labels]
    integ = [S.get(k, {}).get("integrity_violated_pct", np.nan) for k in labels]
    a3.bar(
        x - w / 2 - 0.01,
        sub,
        w,
        color=GREEN,
        label="submitted (task_complete)",
        zorder=3,
    )
    a3.bar(
        x + w / 2 + 0.01,
        integ,
        w,
        color=ORANGE,
        label="protected file changed",
        zorder=3,
    )
    for xi, v in zip(x - w / 2 - 0.01, sub):
        if v == v:
            a3.text(xi, v + 1.5, f"{v:.0f}", ha="center", fontsize=8, color=INK)
    for xi, v in zip(x + w / 2 + 0.01, integ):
        if v == v:
            a3.text(xi, v + 1.5, f"{v:.0f}", ha="center", fontsize=8, color=INK)
    a3.set_xticks(x)
    a3.set_xticklabels(labels, fontsize=9, color=INK)
    a3.set_ylim(0, 105)
    a3.legend(frameon=False, fontsize=8, loc="upper right")
    a3.set_title(
        "Guards from the sandbox (no judge)", fontsize=10, color=INK, loc="left"
    )

    fig.suptitle(a.title, fontsize=11, color=INK, x=0.01, ha="left")
    fig.tight_layout()
    a.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, facecolor=SURF)
    print("saved", a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
