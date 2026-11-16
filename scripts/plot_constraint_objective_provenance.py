#!/usr/bin/env python
# ABOUTME: One slide showing that our constraint loss is Malik et al. 2021's maximum-likelihood update in discriminative form, and naming the three deliberate departures
# ABOUTME: Run: python scripts/plot_constraint_objective_provenance.py
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

S, INK, INK2, MUTED, LINE = "#fcfcfb", "#1c1b19", "#57534e", "#8a857e", "#d5d1cb"
BLUE, BLUE_BG, VIOL, VIOL_BG = "#2563eb", "#eaf1fd", "#be123c", "#fdecef"
MONO = ["DejaVu Sans Mono", "Menlo", "monospace"]

fig, ax = plt.subplots(figsize=(14.4, 7.4))
fig.patch.set_facecolor(S); ax.set_facecolor(S)
ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

ax.text(.5, .975, "Which paper the objective comes from", color=INK,
        fontsize=20, fontweight="bold", ha="center", va="top")
ax.text(.5, .930, "Malik et al., ICML 2021 — Inverse Constrained Reinforcement Learning  (arXiv:2011.09999)",
        color=INK2, fontsize=12, ha="center", va="top")

# --- the paper's gradient -------------------------------------------------
ax.add_patch(FancyBboxPatch((.03, .615), .94, .230,
    boxstyle="round,pad=0,rounding_size=.010", facecolor=S,
    edgecolor=LINE, linewidth=1.6))
ax.text(.05, .820, "THE PAPER'S UPDATE", color=MUTED, fontsize=10.5,
        fontweight="bold", va="top")
ax.text(.5, .762,
        "∇ log p(expert)   =   Σ  ∇ log φ(s,a)        −        E      [ ∇ log φ(s,a) ]",
        color=INK, fontsize=17, ha="center", va="center", family=MONO)
ax.text(.383, .733, "over expert traces", color=BLUE, fontsize=11, ha="center", va="top")
ax.text(.735, .733, "over the current policy", color=VIOL, fontsize=11, ha="center", va="top")
ax.text(.383, .695, "push feasibility UP", color=BLUE, fontsize=12.5,
        ha="center", va="top", fontweight="bold")
ax.text(.735, .695, "push feasibility DOWN", color=VIOL, fontsize=12.5,
        ha="center", va="top", fontweight="bold")
ax.text(.5, .650, "Two terms, opposite directions. That is the whole method.",
        color=INK2, fontsize=11.5, ha="center", va="top")

ax.text(.5, .592, "with a labelled negative set and one score per trajectory, this is exactly …",
        color=MUTED, fontsize=11, ha="center", va="top", style="italic")

# --- our loss -------------------------------------------------------------
ax.add_patch(FancyBboxPatch((.03, .368), .94, .185,
    boxstyle="round,pad=0,rounding_size=.010", facecolor=BLUE_BG,
    edgecolor=BLUE, linewidth=1.8))
ax.text(.05, .530, "WHAT WE TRAIN", color=BLUE, fontsize=10.5,
        fontweight="bold", va="top")
ax.text(.5, .462, "L(θ)   =   − Σ log(1 − C(τ))        −        Σ log C(τ)",
        color=INK, fontsize=17, ha="center", va="center", family=MONO)
ax.text(.352, .428, "expert traces", color=BLUE, fontsize=11, ha="center", va="top")
ax.text(.700, .428, "unsafe + on-policy rollouts", color=VIOL, fontsize=11,
        ha="center", va="top")
ax.text(.5, .392, "Binary cross-entropy — the same two terms, the same directions.",
        color=INK, fontsize=12, ha="center", va="top", fontweight="bold")

# --- the fix + departures -------------------------------------------------
ax.add_patch(FancyBboxPatch((.03, .168), .455, .175,
    boxstyle="round,pad=0,rounding_size=.010", facecolor=VIOL_BG,
    edgecolor=VIOL, linewidth=1.6))
ax.text(.048, .322, "THIS WEEK'S FIX", color=VIOL, fontsize=10.5,
        fontweight="bold", va="top")
ax.text(.048, .290,
        "The paper takes the second term over the CURRENT\n"
        "policy. We had used a frozen off-policy set.\n"
        "Adding on-policy rollouts:  F1 0.730 → 0.847",
        color=INK, fontsize=11, va="top", linespacing=1.7)

ax.add_patch(FancyBboxPatch((.515, .168), .455, .175,
    boxstyle="round,pad=0,rounding_size=.010", facecolor=S,
    edgecolor=LINE, linewidth=1.6))
ax.text(.533, .322, "THREE DELIBERATE DEPARTURES", color=MUTED, fontsize=10.5,
        fontweight="bold", va="top")
ax.text(.533, .290,
        "1.  one score per episode, not per step\n"
        "2.  no minimality regulariser\n"
        "3.  a point estimate, not a Beta posterior (VICRL)",
        color=INK2, fontsize=11, va="top", linespacing=1.7)

# --- footer ---------------------------------------------------------------
ax.text(.5, .120,
        "We also implemented the per-step version and PUCL (Peng & Billard, RA-L 2024). Both scored worse here.",
        color=INK2, fontsize=11.5, ha="center", va="top")
ax.text(.5, .072,
        "Liu et al., ICLR 2023 (arXiv:2206.09670) · Subramanian et al., ICML 2024 (arXiv:2406.16782) · "
        "Peng & Billard, RA-L 2024 (arXiv:2408.01622) · survey arXiv:2409.07569",
        color=MUTED, fontsize=9.5, ha="center", va="top")

fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
out = Path("output/plots/constraint_objective_provenance.png")
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, dpi=200, facecolor=S)
print(f"wrote {out}")
