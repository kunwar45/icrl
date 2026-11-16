#!/usr/bin/env python
# ABOUTME: Slide-ready flow diagram of how the ODCV constraint function is trained: where the labels come from, what text the model reads, the encoder and head, the loss, and the scenario-disjoint split
# ABOUTME: Run: python scripts/plot_constraint_training_flow_diagram.py --out output/plots/constraint_training_flow.png
from __future__ import annotations
import argparse
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

S, INK, INK2, MUTED, LINE = "#fcfcfb", "#1c1b19", "#57534e", "#8a857e", "#d9d5cf"
SAFE, VIOL, NEUT = "#2563eb", "#be123c", "#a16207"
SAFE_BG, VIOL_BG, NEUT_BG, GREY_BG = "#eaf1fd", "#fdecef", "#faf1e0", "#f2efea"
MONO = ["DejaVu Sans Mono", "Menlo", "monospace"]


def box(ax, x, y, w, h, fc, ec, lw=1.2, r=0.010):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
        boxstyle=f"round,pad=0,rounding_size={r}", facecolor=fc,
        edgecolor=ec, linewidth=lw, zorder=2))


def arrow(ax, x0, y0, x1, y1, c=MUTED, lw=1.4, style="-|>"):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle=style,
        mutation_scale=12, color=c, linewidth=lw, shrinkA=0, shrinkB=0, zorder=3))


def title(ax, x, y, t, c=INK, s=10.5):
    ax.text(x, y, t, color=c, fontsize=s, fontweight="bold", va="top")


def body(ax, x, y, t, c=INK2, s=9.0, mono=False, lh=1.55):
    ax.text(x, y, t, color=c, fontsize=s, va="top", linespacing=lh,
            family=MONO if mono else None)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path,
                    default=Path("output/plots/constraint_training_flow.png"))
    a = ap.parse_args()
    fig, ax = plt.subplots(figsize=(14.6, 8.2))
    fig.patch.set_facecolor(S); ax.set_facecolor(S)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    ax.text(0, .985, "How the constraint function is trained", color=INK,
            fontsize=16, fontweight="bold", va="top")
    ax.text(0, .945, "Contrast one policy's own aligned and misaligned behaviour, "
            "then score a whole trajectory for whether it violated the scenario's rule.",
            color=INK2, fontsize=10.5, va="top")

    # ---------- STAGE 1: where trajectories come from -----------------------
    y0 = .885
    box(ax, 0, y0 - .155, .30, .155, GREY_BG, LINE)
    title(ax, .014, y0 - .014, "1 · ONE POLICY, SAMPLED", MUTED, 9.2)
    body(ax, .014, y0 - .042,
         "Qwen3.6-27B + adapter\n(no safety training)\ntemperature 0.7, same prompts")
    body(ax, .014, y0 - .118, "Sample repeatedly → it does\ndifferent things.", INK, 9.0)

    # ---------- STAGE 2: labelling ------------------------------------------
    box(ax, .345, y0 - .155, .30, .155, GREY_BG, LINE)
    title(ax, .359, y0 - .014, "2 · TWO INDEPENDENT LABELS", MUTED, 9.2)
    body(ax, .359, y0 - .042,
         "LLM judge scores 0–5 on a rubric\n  ≥3 = misaligned\nSeparate auditor re-reads it and\n  assigns severity + evidence")
    body(ax, .359, y0 - .128, "Keep only where they agree.", INK, 9.0)
    arrow(ax, .30, y0 - .078, .345, y0 - .078)

    # ---------- STAGE 3: the two sets ---------------------------------------
    box(ax, .69, y0 - .155, .31, .155, S, LINE)
    title(ax, .704, y0 - .014, "3 · THE CONTRAST SET", MUTED, 9.2)
    box(ax, .704, y0 - .105, .137, .052, SAFE_BG, SAFE, lw=1.1)
    ax.text(.7725, y0 - .062, "EXPERT", color=SAFE, fontsize=9.4,
            fontweight="bold", ha="center", va="top")
    ax.text(.7725, y0 - .085, "232 · severity 0–1", color=INK2, fontsize=8.2,
            ha="center", va="top")
    box(ax, .855, y0 - .105, .137, .052, VIOL_BG, VIOL, lw=1.1)
    ax.text(.9235, y0 - .062, "UNSAFE", color=VIOL, fontsize=9.4,
            fontweight="bold", ha="center", va="top")
    ax.text(.9235, y0 - .085, "294 · severity 3–5", color=INK2, fontsize=8.2,
            ha="center", va="top")
    body(ax, .704, y0 - .124, "The sets never overlap in severity.", INK, 8.6)
    arrow(ax, .645, y0 - .078, .69, y0 - .078)

    # ---------- the on-policy pool ------------------------------------------
    y1 = .690
    box(ax, .345, y1 - .088, .655, .088, SAFE_BG, SAFE, lw=1.3)
    title(ax, .359, y1 - .013, "+ 592 JUDGE-LABELLED ON-POLICY ROLLOUTS", SAFE, 9.2)
    body(ax, .359, y1 - .040,
         "Rollouts the policy produced during fine-tuning, labelled by the same judges. Canonical ICRL takes its\n"
         "negative expectation over the CURRENT policy — using a frozen set instead was our biggest single mistake.",
         INK2, 8.8)
    arrow(ax, .845, y0 - .155, .845, y1)

    # ---------- STAGE 4: input rendering ------------------------------------
    y2 = .552
    box(ax, 0, y2 - .200, .455, .200, S, LINE)
    title(ax, .014, y2 - .013, "4 · WHAT THE MODEL READS", MUTED, 9.2)
    box(ax, .014, y2 - .188, .427, .160, GREY_BG, LINE, lw=1.0)
    body(ax, .026, y2 - .044,
         "[SCENARIO]\n  “…performance is evaluated by\n   validate_submission.sh”\n"
         "[TRAJECTORY]\n  [THOUGHT] adjust p-value to meet threshold\n"
         "  [ACTION]  python3: data['p'] = data['p']/2\n  [ACTION]  task_complete: validated",
         INK, 8.2, mono=True, lh=1.5)
    ax.text(.014, y2 - .215, "The scenario text is what makes an action a violation.",
            color=VIOL, fontsize=9.0, va="top", fontweight="bold")
    arrow(ax, .17, y1 - .088, .17, y2)

    # ---------- STAGE 5: the model ------------------------------------------
    box(ax, .505, y2 - .200, .245, .200, S, LINE)
    title(ax, .519, y2 - .013, "5 · THE MODEL", MUTED, 9.2)
    for i, (t, sub) in enumerate((
            ("Qwen2.5-7B-Instruct", "+ LoRA r=16 on q,k,v,o"),
            ("mean-pool 3072 tokens", "→ ℝ³⁵⁸⁴"),
            ("MLP 3584→256→1", "sigmoid"))):
        yy = y2 - .048 - i * .052
        box(ax, .519, yy - .038, .217, .038, GREY_BG, LINE, lw=1.0)
        ax.text(.5275, yy - .012, t, color=INK, fontsize=8.8, va="top")
        ax.text(.5275, yy - .028, sub, color=MUTED, fontsize=7.8, va="top")
        if i < 2:
            arrow(ax, .6275, yy - .038, .6275, yy - .052, LINE, lw=1.1)
    ax.text(.519, y2 - .215, "C_θ(τ) ∈ (0,1) — one score per trajectory",
            color=INK, fontsize=9.0, va="top", fontweight="bold")
    arrow(ax, .455, y2 - .100, .505, y2 - .100)

    # ---------- STAGE 6: loss + split ---------------------------------------
    box(ax, .795, y2 - .200, .205, .200, NEUT_BG, NEUT, lw=1.2)
    title(ax, .809, y2 - .013, "6 · THE LOSS", NEUT, 9.2)
    ax.text(.809, y2 - .048, "Binary cross-entropy", color=INK, fontsize=9.2, va="top")
    ax.text(.809, y2 - .075,
            "−Σ log(1−C)   over expert\n−Σ log C        over unsafe",
            color=INK2, fontsize=8.4, va="top", family=MONO, linespacing=1.7)
    ax.text(.809, y2 - .132,
            "This IS the canonical\nmaximum-likelihood update:\nraise expert feasibility,\nlower the policy's.",
            color=INK2, fontsize=8.4, va="top", linespacing=1.6)
    arrow(ax, .75, y2 - .100, .795, y2 - .100)

    # ---------- STAGE 7: the split ------------------------------------------
    y3 = .300
    box(ax, 0, y3 - .130, 1.0, .130, S, LINE)
    title(ax, .014, y3 - .013, "7 · SPLIT BY SCENARIO — nothing about a test scenario is ever seen", MUTED, 9.2)
    cols = [(.014, "TRAIN", "796 trajectories", "fits the LoRA and the head", INK2, GREY_BG),
            (.345, "VALIDATION", "198 trajectories", "picks the epoch AND the threshold", NEUT, NEUT_BG),
            (.676, "HELD-OUT TEST", "186 rollouts, 10 scenarios", "reported once, never tuned on", SAFE, SAFE_BG)]
    for x, name, n, note, c, bg in cols:
        box(ax, x, y3 - .118, .310, .070, bg, c, lw=1.1)
        ax.text(x + .012, y3 - .060, name, color=c, fontsize=9.4, fontweight="bold", va="top")
        ax.text(x + .012, y3 - .081, n, color=INK, fontsize=8.6, va="top")
        ax.text(x + .012, y3 - .100, note, color=MUTED, fontsize=8.2, va="top")
    arrow(ax, .30, y2 - .200, .30, y3)

    # ---------- result ------------------------------------------------------
    y4 = .135
    box(ax, 0, y4 - .125, 1.0, .125, SAFE_BG, SAFE, lw=1.4)
    ax.text(.016, y4 - .014, "RESULT", color=SAFE, fontsize=9.2,
            fontweight="bold", va="top")
    ax.text(.016, y4 - .046, "F1 0.847 ± 0.017", color=SAFE, fontsize=19,
            fontweight="bold", va="top")
    ax.text(.016, y4 - .098, "four seeds, held-out scenarios", color=INK2,
            fontsize=8.6, va="top")
    ax.text(.30, y4 - .046, "0.828", color=INK, fontsize=19, fontweight="bold", va="top")
    ax.text(.30, y4 - .098, "what two 70B-class judges score\nagainst each other on the same rollouts",
            color=INK2, fontsize=8.6, va="top", linespacing=1.5)
    ax.text(.585, y4 - .046, "47% → 84%", color=INK, fontsize=19, fontweight="bold", va="top")
    ax.text(.585, y4 - .098, "purity of the “violation” label\nthe fine-tuning loop trains against",
            color=INK2, fontsize=8.6, va="top", linespacing=1.5)
    ax.text(.845, y4 - .046, "= at the\nlabel ceiling", color=SAFE, fontsize=13,
            fontweight="bold", va="top", linespacing=1.4)
    arrow(ax, .30, y3 - .130, .30, y4)

    fig.tight_layout(pad=0.5)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, dpi=200, facecolor=S)
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
