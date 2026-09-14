#!/usr/bin/env python
# ABOUTME: Plots the ODCV constraint model's precision-recall curves for the three input/data ablation arms against the inter-judge agreement ceiling
# ABOUTME: Run: python scripts/plot_constraint_context_ablation.py --results-dir output/constraint_context --out output/plots/constraint_context_ablation.png
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#1c1b19"
INK_2 = "#57534e"
MUTED = "#8a857e"
GRID = "#e4e1dc"

ARMS = [
    ("actions_only", "Actions only", "#b45309"),
    ("context", "+ scenario + reasoning", "#be123c"),
    ("context_rollouts", "+ on-policy rollouts", "#2563eb"),
]
# Same 186 held-out rollouts: Llama-3.3-70B judging against Mistral-24B.
JUDGE_PRECISION, JUDGE_RECALL, JUDGE_F1 = 0.814, 0.842, 0.828
ZERO_SHOT = "constraint_zero_shot_qwen2.5-7b.json"  # prompted 7B, the untrained floor


def pr_points(scores, labels):
    order = sorted(range(len(scores)), key=lambda i: -scores[i])
    P = sum(labels)
    tp = 0
    rec, prec = [0.0], [1.0]
    for rank, i in enumerate(order, 1):
        tp += labels[i]
        rec.append(tp / P)
        prec.append(tp / rank)
    return rec, prec


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results-dir", type=Path, default=Path("output/constraint_context"))
    ap.add_argument("--out", type=Path, default=Path("output/plots/constraint_context_ablation.png"))
    a = ap.parse_args()

    data = {
        k: json.loads((a.results_dir / f"constraint_context_{k}.json").read_text())
        for k, _, _ in ARMS
    }
    prevalence = data["context_rollouts"]["test"]["prevalence"]
    zs_path = a.results_dir / ZERO_SHOT
    zs = json.loads(zs_path.read_text())["held_out"] if zs_path.exists() else None

    fig, (ax, bx) = plt.subplots(
        1, 2, figsize=(11.6, 4.9), gridspec_kw={"width_ratios": [1.42, 1.0]}
    )
    fig.patch.set_facecolor(SURFACE)

    # ---- A: precision-recall ------------------------------------------------
    ax.set_facecolor(SURFACE)
    ax.axhline(prevalence, color=MUTED, lw=1.2, ls=(0, (4, 3)), zorder=1)
    ax.text(
        0.012, prevalence + 0.014, f"chance ({prevalence:.0%} of rollouts violate)",
        color=MUTED, fontsize=8.5, va="bottom",
    )
    for key, label, colour in ARMS:
        d = data[key]
        rec, prec = pr_points(d["scores"], d["labels"])
        ax.plot(rec, prec, color=colour, lw=2.0, zorder=3,
                solid_capstyle="round", label=label)
        b = d["test"]["best_f1"]
        ax.plot(
            b["recall"], b["precision"], "o", ms=8.5, color=colour,
            mec=SURFACE, mew=2.0, zorder=5,
        )
    if zs:
        z = zs["at_rubric_threshold_3"]
        ax.plot(
            z["recall"], z["precision"], "^", ms=9.0, color=MUTED,
            mec=SURFACE, mew=1.6, zorder=5,
        )
        ax.annotate(
            "prompted 7B, untrained",
            (z["recall"], z["precision"]), xytext=(z["recall"] - 0.045, z["precision"] - 0.055),
            color=MUTED, fontsize=8.8, ha="right", va="top",
        )

    # Hollow, so the best arm's own point stays visible underneath: they coincide.
    ax.plot(
        JUDGE_RECALL, JUDGE_PRECISION, "D", ms=14.0, mfc="none",
        mec=INK, mew=1.7, zorder=6,
    )
    ax.annotate(
        "two judges agreeing\nwith each other",
        (JUDGE_RECALL, JUDGE_PRECISION + 0.02), xytext=(JUDGE_RECALL, 1.0),
        color=INK, fontsize=9.0, ha="center", va="top",
        arrowprops=dict(arrowstyle="-", color=INK, lw=0.9, shrinkA=3, shrinkB=6),
    )
    leg = ax.legend(
        loc="lower left", frameon=False, fontsize=9.4,
        handlelength=1.5, borderpad=0.2, labelspacing=0.45,
        bbox_to_anchor=(0.005, 0.015),
    )
    for t in leg.get_texts():
        t.set_color(INK_2)
    ax.text(
        0.005, 0.225, "dot = each arm's best operating point",
        transform=ax.transAxes, color=MUTED, fontsize=8.5,
    )

    ax.set_xlim(0, 1.005)
    ax.set_ylim(0, 1.07)
    ax.set_xlabel("Recall — violations caught", color=INK_2, fontsize=10)
    ax.set_ylabel("Precision — flags that are real", color=INK_2, fontsize=10)
    ax.set_title(
        "Constraint model on 10 held-out scenarios (186 rollouts)",
        color=INK, fontsize=11.5, fontweight="600", loc="left", pad=10,
    )
    ax.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=9)

    # ---- B: the numbers -----------------------------------------------------
    bx.set_facecolor(SURFACE)
    bx.axis("off")
    rows = [
        (label, data[k]["test"]) for k, label, _ in ARMS
    ]
    y = 0.93
    bx.text(0.0, y, "At each arm's best operating point", color=INK,
            fontsize=11.5, fontweight="600", va="top")
    y -= 0.115
    hdr = ["", "AUROC", "Prec.", "Recall", "F1"]
    xs = [0.0, 0.545, 0.685, 0.825, 0.975]
    for x, h in zip(xs, hdr):
        bx.text(x, y, h, color=MUTED, fontsize=9.0, va="top",
                ha="left" if x == 0.0 else "right")
    y -= 0.055
    bx.plot([0.0, 1.0], [y, y], color=GRID, lw=1.0)
    for (label, t), (_, _, colour) in zip(rows, ARMS):
        y -= 0.105
        b = t["best_f1"]
        bx.plot([-0.035], [y - 0.018], marker="s", ms=7, color=colour, clip_on=False)
        cells = [
            label,
            f"{t['auroc']:.3f}",
            f"{b['precision']:.0%}",
            f"{b['recall']:.0%}",
            f"{b['f1']:.3f}",
        ]
        for x, c, h in zip(xs, cells, hdr):
            bx.text(
                x, y, c, color=INK if h else INK_2, fontsize=9.3, va="top",
                ha="left" if x == 0.0 else "right",
                fontweight="700" if (h and label.startswith("+ on-policy")) else "400",
            )
    y -= 0.075
    bx.plot([0.0, 1.0], [y, y], color=GRID, lw=1.0)
    y -= 0.105
    bx.plot([-0.035], [y - 0.018], marker="D", ms=7, color=INK, clip_on=False)
    for x, c, h in zip(
        xs, ["Judge vs judge", "\u2014", f"{JUDGE_PRECISION:.0%}",
             f"{JUDGE_RECALL:.0%}", f"{JUDGE_F1:.3f}"], hdr
    ):
        bx.text(x, y, c, color=INK, fontsize=9.3, va="top",
                ha="left" if x == 0.0 else "right")

    if zs:
        z = zs["at_rubric_threshold_3"]
        y -= 0.105
        bx.plot([-0.035], [y - 0.018], marker="^", ms=7, color=MUTED, clip_on=False)
        zf1 = 2 * z["precision"] * z["recall"] / max(z["precision"] + z["recall"], 1e-9)
        for x, c in zip(
            xs, ["Prompted 7B, untrained", f"{zs['auroc']:.3f}",
                 f"{z['precision']:.0%}", f"{z['recall']:.0%}", f"{zf1:.3f}"]
        ):
            bx.text(x, y, c, color=INK_2, fontsize=9.3, va="top",
                    ha="left" if x == 0.0 else "right")

    b = data["context_rollouts"]["test"]["best_f1"]
    y -= 0.115
    bx.text(0.0, y, "Confusion matrix, + on-policy rollouts", color=INK,
            fontsize=10.4, fontweight="600", va="top")
    y -= 0.085
    for x, h in zip([0.60, 0.90], ["called safe", "called violation"]):
        bx.text(x, y, h, color=MUTED, fontsize=9.0, va="top", ha="right")
    for lbl, (a_, b_), good in (
        ("actually safe", (b["tn"], b["fp"]), (True, False)),
        ("actually violating", (b["fn"], b["tp"]), (False, True)),
    ):
        y -= 0.085
        bx.text(0.0, y, lbl, color=INK_2, fontsize=9.4, va="top")
        for x, v, g in zip([0.60, 0.90], (a_, b_), good):
            bx.text(x, y, str(v), color=INK if g else "#be123c",
                    fontsize=10.2, va="top", ha="right", fontweight="700")
    bx.set_xlim(-0.06, 1.0)
    bx.set_ylim(0, 1.0)

    fig.tight_layout()
    a.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, dpi=200, facecolor=SURFACE)
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
