#!/usr/bin/env python
# ABOUTME: Three-panel summary of the ODCV constraint result for a talk: the confusion matrix at the deployed threshold, F1 by variant against the inter-judge ceiling, and the stress tests before and after
# ABOUTME: Run: python scripts/plot_constraint_summary_for_review.py --out output/plots/constraint_summary.png
from __future__ import annotations
import argparse
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

S, INK, INK2, MUTED, GRID = "#fcfcfb", "#1c1b19", "#57534e", "#8a857e", "#e4e1dc"
OLD, NEW, BAD = "#b45309", "#2563eb", "#be123c"

CM = {"tn": 117, "fp": 10, "fn": 11, "tp": 48}      # deployed threshold 0.686
ARMS = [("Prompted 7B\n(untrained)", 0.504, MUTED),
        ("Actions only\n(old setup)", 0.655, OLD),
        ("+ scenario\n+ reasoning", 0.730, BAD),
        ("+ on-policy\nrollouts", 0.847, NEW)]
JUDGE = 0.828
STRESS = [("Within-scenario\nranking", 0.67, 0.913),
          ("AUROC within\nlength bands", 0.78, 0.916),
          ("Calibration\n(1 - ECE)", 0.65, 0.926)]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=Path("output/plots/constraint_summary.png"))
    a = ap.parse_args()
    fig, (ax, bx, cx) = plt.subplots(1, 3, figsize=(15.4, 4.9),
                                     gridspec_kw={"width_ratios": [1.0, 1.15, 1.0]})
    fig.patch.set_facecolor(S)

    # --- A: confusion matrix ------------------------------------------------
    ax.set_facecolor(S); ax.axis("off")
    ax.set_title("Constraint vs judge verdict\n186 held-out rollouts, 10 unseen scenarios",
                 color=INK, fontsize=12, fontweight="600", loc="left", pad=14)
    cells = [((0, 1), CM["tn"], "correct"), ((1, 1), CM["fp"], "false alarm"),
             ((0, 0), CM["fn"], "missed"),  ((1, 0), CM["tp"], "correct")]
    for (cx_, cy_), v, kind in cells:
        good = kind == "correct"
        ax.add_patch(plt.Rectangle((cx_ * .5 + .02, cy_ * .5 + .06), .46, .40,
                                   facecolor="#eaf1fd" if good else "#fdecef",
                                   edgecolor=NEW if good else BAD, lw=1.4))
        ax.text(cx_ * .5 + .25, cy_ * .5 + .33, str(v), ha="center", va="center",
                fontsize=30, fontweight="bold", color=NEW if good else BAD)
        ax.text(cx_ * .5 + .25, cy_ * .5 + .15, kind, ha="center", va="center",
                fontsize=10, color=INK2)
    ax.text(.25, 1.0, "called SAFE", ha="center", fontsize=10.5, color=MUTED)
    ax.text(.75, 1.0, "called VIOLATION", ha="center", fontsize=10.5, color=MUTED)
    ax.text(-.02, .78, "actually\nsafe", ha="right", va="center", fontsize=10.5, color=MUTED)
    ax.text(-.02, .28, "actually\nviolating", ha="right", va="center", fontsize=10.5, color=MUTED)
    ax.text(.5, -.10, "precision 82.8%   ·   recall 81.4%   ·   accuracy 88.7%",
            ha="center", fontsize=11.5, color=INK, fontweight="600")
    ax.text(.5, -.19, "threshold fixed on validation, applied blind", ha="center",
            fontsize=9.5, color=MUTED)
    ax.set_xlim(-.2, 1.02); ax.set_ylim(-.24, 1.06)

    # --- B: F1 by variant ---------------------------------------------------
    bx.set_facecolor(S)
    xs = range(len(ARMS))
    bx.bar(xs, [v for _, v, _ in ARMS], color=[c for _, _, c in ARMS], width=.62,
           zorder=3)
    bx.errorbar([3], [0.847], yerr=[0.017], color=INK, capsize=5, lw=1.4, zorder=5, fmt="none")
    bx.axhline(JUDGE, color=INK, lw=1.4, ls=(0, (5, 3)), zorder=4)
    bx.text(len(ARMS) - .5, JUDGE + .012, "two judges agreeing with each other  (0.828)",
            ha="right", fontsize=10, color=INK)
    for i, (_, v, _) in enumerate(ARMS):
        bx.text(i, v + .022, f"{v:.3f}", ha="center", fontsize=11, fontweight="bold", color=INK)
    bx.set_xticks(list(xs)); bx.set_xticklabels([n for n, _, _ in ARMS], fontsize=9.5, color=INK2)
    bx.set_ylim(0, 1.0); bx.set_ylabel("F1 against the judge verdict", color=INK2, fontsize=10.5)
    bx.set_title("Every lever, same test set", color=INK, fontsize=12,
                 fontweight="600", loc="left", pad=14)
    bx.grid(axis="y", color=GRID, lw=.8); bx.set_axisbelow(True)
    for s in ("top", "right"): bx.spines[s].set_visible(False)
    for s in ("left", "bottom"): bx.spines[s].set_color(GRID)
    bx.tick_params(colors=INK2, labelsize=9.5)

    # --- C: stress tests ----------------------------------------------------
    cx.set_facecolor(S)
    ys = range(len(STRESS))
    h = .30
    cx.barh([y + h / 2 for y in ys], [o for _, o, _ in STRESS], height=h,
            color=OLD, zorder=3, label="old constraint")
    cx.barh([y - h / 2 for y in ys], [n for _, _, n in STRESS], height=h,
            color=NEW, zorder=3, label="new constraint")
    for y, (_, o, n) in zip(ys, STRESS):
        cx.text(o + .015, y + h / 2, f"{o:.2f}", va="center", fontsize=10, color=OLD)
        cx.text(n + .015, y - h / 2, f"{n:.2f}", va="center", fontsize=10,
                color=NEW, fontweight="bold")
    cx.set_yticks(list(ys)); cx.set_yticklabels([n for n, _, _ in STRESS],
                                                fontsize=9.5, color=INK2)
    cx.invert_yaxis(); cx.set_xlim(0, 1.12)
    cx.set_title("Stress tests it used to fail", color=INK, fontsize=12,
                 fontweight="600", loc="left", pad=14)
    cx.legend(frameon=False, fontsize=10, loc="upper center",
              bbox_to_anchor=(.5, -.07), ncol=2)
    cx.grid(axis="x", color=GRID, lw=.8); cx.set_axisbelow(True)
    for s in ("top", "right"): cx.spines[s].set_visible(False)
    for s in ("left", "bottom"): cx.spines[s].set_color(GRID)
    cx.tick_params(colors=INK2, labelsize=9.5)

    fig.tight_layout(pad=1.4)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, dpi=200, facecolor=S)
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
