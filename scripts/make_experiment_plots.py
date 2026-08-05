#!/usr/bin/env python3
"""
Turn a run's artifacts into figures and a single self-contained report.

Reads whatever the run produced and skips what it didn't, so it is safe to call
on a partial run — a job that died during fine-tuning still gets its constraint
and gate figures.

Inputs (all optional):
    <logs>/<run>/<run>_constraint_metrics.jsonl   ICRL training curve
    <logs>/<run>/<run>_finetune_metrics.jsonl     Lagrangian dynamics
    <ckpt>/<run>/held_out_metrics.json            gate: AUROC, ROC, score spread
    <ckpt>/<run>/train_metrics.json               train-split metrics
    <ckpt>/<run>_eval_base/cup_eval.json          baseline CuP
    <ckpt>/<run>_eval_tuned/cup_eval.json         tuned CuP
    <logs>/<run>_experiment.json                  per-stage status and timings

Outputs:
    <logs>/<run>/plots/*.png  (+ *.pdf with --pdf)
    <logs>/<run>/plots/report.html   one file, images inlined — scp it off the
                                     cluster and open it anywhere

Usage:
    python scripts/make_experiment_plots.py --run-name icrl_cluster
    python scripts/make_experiment_plots.py --run-name icrl_cluster --theme dark --pdf
"""
from __future__ import annotations

import argparse
import base64
import html
import json
import sys
from pathlib import Path
from typing import Any, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

from src.utils.viz import (  # noqa: E402
    THEMES, Theme, apply_theme, category_limits, empty_note, end_marker,
    fit_bar_width, integer_axis, label_endpoints, label_room, legend,
    reference_line, rounded_bar, style_axes, titled,
)


# ── Artifact loading ──────────────────────────────────────────────────────────

def read_json(path: Path) -> Optional[dict]:
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    if not path.exists():
        return rows
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


class RunArtifacts:
    """Everything a run may have written, with `None` where it didn't."""

    def __init__(self, run_name: str, log_dir: Path, ckpt_dir: Path):
        self.run_name = run_name
        run_logs = log_dir / run_name

        rows = read_jsonl(run_logs / f"{run_name}_constraint_metrics.jsonl")
        self.constraint_steps = [r for r in rows if "policy_constraint_score" in r]
        self.constraint_evals = [r for r in rows if "eval_auroc" in r]
        self.finetune_steps = read_jsonl(run_logs / f"{run_name}_finetune_metrics.jsonl")

        self.gate = read_json(ckpt_dir / run_name / "held_out_metrics.json")
        self.train_metrics = read_json(ckpt_dir / run_name / "train_metrics.json")
        self.eval_base = read_json(ckpt_dir / f"{run_name}_eval_base" / "cup_eval.json")
        self.eval_tuned = read_json(ckpt_dir / f"{run_name}_eval_tuned" / "cup_eval.json")
        self.experiment = read_json(log_dir / f"{run_name}_experiment.json")

    def summary(self, which: str) -> Optional[dict]:
        blob = getattr(self, f"eval_{which}")
        return (blob or {}).get("summary")

    @property
    def anything(self) -> bool:
        return any([self.constraint_steps, self.finetune_steps, self.gate,
                    self.eval_base, self.eval_tuned, self.experiment])


# ── Figures ───────────────────────────────────────────────────────────────────

def fig_constraint_training(art: RunArtifacts, theme: Theme) -> Optional[plt.Figure]:
    """
    Two panels sharing the iteration axis.

    Separate panels rather than one plot with two y-scales: constraint score and
    AUROC are different measures, and overlaying them on twin axes would invent
    a relationship out of the arbitrary alignment of the two scales.
    """
    if not art.constraint_steps:
        return None

    steps = [r["step"] for r in art.constraint_steps]
    policy = [r["policy_constraint_score"] for r in art.constraint_steps]
    expert = [r["expert_constraint_score"] for r in art.constraint_steps]

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(7.4, 6.2), sharex=True,
        gridspec_kw={"height_ratios": [1.35, 1], "hspace": 0.32},
    )

    ax1.plot(steps, policy, color=theme.series[0], label="Policy (unsafe) demos", zorder=4)
    ax1.plot(steps, expert, color=theme.series[1], label="Expert (safe) demos", zorder=4)
    end_marker(ax1, steps[-1], policy[-1], theme.series[0], theme)
    end_marker(ax1, steps[-1], expert[-1], theme.series[1], theme)
    reference_line(ax1, 0.8, "β = 0.80  expert anchor", theme)

    ax1.set_ylabel("C$_θ$(τ)")
    ax1.set_ylim(-0.03, 1.06)
    label_room(ax1)
    label_endpoints(ax1, [(steps[-1], policy[-1], f"{policy[-1]:.2f}"),
                          (steps[-1], expert[-1], f"{expert[-1]:.2f}")], theme)
    style_axes(ax1, theme)
    titled(ax1, "Constraint learning",
           "C$_θ$ should push policy scores up while expert scores stay under β",
           theme)
    legend(ax1, theme, loc="lower right")

    if art.constraint_evals:
        e_steps = [r["step"] for r in art.constraint_evals]
        auroc = [r["eval_auroc"] for r in art.constraint_evals]
        ax2.plot(e_steps, auroc, color=theme.series[0], marker="o", markersize=8,
                 markeredgecolor=theme.surface, markeredgewidth=2.0, zorder=4)
        reference_line(ax2, 0.75, "gate = 0.75", theme)
        ax2.set_ylim(0.35, 1.04)
        ax2.set_ylabel("AUROC")
        label_endpoints(ax2, [(e_steps[-1], auroc[-1], f"{auroc[-1]:.3f}")], theme)
        titled(ax2, "Separation during training",
               "train-pool AUROC, unsafe scored higher than safe", theme)
    else:
        empty_note(ax2, "No periodic evaluations recorded\n"
                        "(constraint.training.eval_every)", theme)

    style_axes(ax2, theme)
    integer_axis(ax2)
    ax2.set_xlabel("ICRL iteration")
    return fig


def fig_gate(art: RunArtifacts, theme: Theme) -> Optional[plt.Figure]:
    """Held-out score distributions and the ROC they imply."""
    gate = art.gate
    if not gate:
        return None

    safe = np.asarray(gate.get("safe_scores") or [], dtype=float)
    unsafe = np.asarray(gate.get("unsafe_scores") or [], dtype=float)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.4, 4.1),
                                   gridspec_kw={"wspace": 0.28})

    if safe.size and unsafe.size:
        bins = np.linspace(0, 1, 21)
        # 2px surface gap between adjacent bars is what separates them — no strokes.
        ax1.hist([safe, unsafe], bins=bins, label=["Safe (held out)", "Unsafe (held out)"],
                 color=[theme.series[1], theme.series[0]], rwidth=0.8, zorder=3)
        ax1.set_xlabel("C$_θ$(τ)   →  higher = judged less safe")
        ax1.set_ylabel("Trajectories")
        integer_axis(ax1, "y")
        # Headroom so the legend never sits on top of the tallest bar.
        ax1.set_ylim(0, max(ax1.get_ylim()[1] * 1.28, 1))
        titled(ax1, "Held-out score distribution",
               f"{int(gate.get('n_safe', safe.size))} safe · "
               f"{int(gate.get('n_unsafe', unsafe.size))} unsafe", theme)
        legend(ax1, theme, loc="upper left")
        style_axes(ax1, theme)

        # ROC — one series, so no legend box; the title names it.
        scores = np.concatenate([safe, unsafe])
        labels = np.concatenate([np.zeros(safe.size), np.ones(unsafe.size)])
        order = np.argsort(-scores)
        labels = labels[order]
        tpr = np.concatenate([[0], np.cumsum(labels) / max(labels.sum(), 1)])
        fpr = np.concatenate([[0], np.cumsum(1 - labels) / max((1 - labels).sum(), 1)])

        ax2.plot([0, 1], [0, 1], color=theme.reference, linewidth=1.0, zorder=2)
        ax2.annotate("chance", xy=(0.62, 0.56), fontsize=8, color=theme.text_muted,
                     rotation=38, ha="center", va="center")
        ax2.plot(fpr, tpr, color=theme.series[0], zorder=4)
        ax2.fill_between(fpr, tpr, color=theme.series[0], alpha=0.10, zorder=3)
        ax2.set_xlim(-0.02, 1.02)
        ax2.set_ylim(-0.02, 1.02)
        ax2.set_xlabel("False positive rate")
        ax2.set_ylabel("True positive rate")
        verdict = "PASS" if gate.get("passed") else "FAIL"
        titled(ax2, f"ROC — AUROC {gate.get('auroc', float('nan')):.3f}  ({verdict})",
               f"gate threshold {gate.get('auroc_gate', 0.75):.2f}", theme)
        style_axes(ax2, theme)
    else:
        empty_note(ax1, "Gate ran before per-trajectory scores were saved —\n"
                        "re-run the gate stage to get distributions", theme)
        empty_note(ax2, f"AUROC {gate.get('auroc', float('nan')):.3f}", theme)

    return fig


def fig_finetune(art: RunArtifacts, theme: Theme) -> Optional[plt.Figure]:
    """
    Four stacked panels sharing the step axis.

    Reward, cost, λ and the rates live on incompatible scales; small multiples
    keep each on its own honest axis instead of a twin-axis overlay.
    """
    rows = art.finetune_steps
    if not rows:
        return None

    steps = [r["step"] for r in rows]

    def series(key: str) -> list[float]:
        return [float(r.get(key, float("nan"))) for r in rows]

    # Generous hspace: each panel carries a title AND a subtitle, which together
    # need more room than matplotlib's default inter-panel gap.
    fig, axes = plt.subplots(4, 1, figsize=(7.4, 10.6), sharex=True,
                             gridspec_kw={"hspace": 0.60})
    ax_r, ax_c, ax_l, ax_p = axes

    reward = series("task_reward")
    ax_r.plot(steps, reward, color=theme.series[0], zorder=4)
    end_marker(ax_r, steps[-1], reward[-1], theme.series[0], theme)
    ax_r.set_ylabel("R(τ)")
    titled(ax_r, "Task reward", "completion, minus step and truncation penalties", theme)
    style_axes(ax_r, theme)
    label_endpoints(ax_r, [(steps[-1], reward[-1], f"{reward[-1]:+.2f}")], theme)

    cost = series("constraint_score")
    ax_c.plot(steps, cost, color=theme.series[0], zorder=4)
    ax_c.fill_between(steps, cost, color=theme.series[0], alpha=0.10, zorder=3)
    end_marker(ax_c, steps[-1], cost[-1], theme.series[0], theme)
    eps = _epsilon(art)
    ax_c.set_ylim(min(0.0, min(cost) * 1.1), max(max(cost), eps or 0) * 1.25 + 0.02)
    if eps is not None:
        reference_line(ax_c, eps, f"ε = {eps:.2f}  budget", theme)
    ax_c.set_ylabel("C$_θ$(τ)")
    titled(ax_c, "Constraint cost vs budget",
           "λ rises while mean cost sits above ε", theme)
    style_axes(ax_c, theme)
    label_endpoints(ax_c, [(steps[-1], cost[-1], f"{cost[-1]:.2f}")], theme)

    lam = series("lambda")
    ax_l.plot(steps, lam, color=theme.series[0], zorder=4)
    end_marker(ax_l, steps[-1], lam[-1], theme.series[0], theme)
    ax_l.set_ylabel("λ")
    titled(ax_l, "Dual variable", "the price the policy pays for constraint cost", theme)
    style_axes(ax_l, theme)
    label_endpoints(ax_l, [(steps[-1], lam[-1], f"{lam[-1]:.3f}")], theme)

    for key, label, colour in (
        ("cup", "CuP", theme.series[0]),
        ("completion_rate", "Completion", theme.series[1]),
        ("violation_rate", "Violation", theme.series[2]),
    ):
        ax_p.plot(steps, series(key), color=colour, label=label, zorder=4)
    ax_p.set_ylim(-0.04, 1.24)   # headroom for the legend
    ax_p.set_ylabel("Rate")
    ax_p.set_xlabel("Fine-tuning step")
    titled(ax_p, "Rollout outcomes", "per batch of training episodes", theme)
    legend(ax_p, theme, loc="upper left", ncol=3)
    style_axes(ax_p, theme)
    integer_axis(ax_p)

    # sharex=True — widening once widens all four; doing it per-axis compounds.
    label_room(ax_p, 0.10)
    return fig


def _epsilon(art: RunArtifacts) -> Optional[float]:
    """Recover the constraint budget from the run report if it recorded one."""
    exp = art.experiment or {}
    eps = exp.get("epsilon")
    return float(eps) if eps is not None else 0.1


def fig_cup(art: RunArtifacts, theme: Theme) -> Optional[plt.Figure]:
    """Baseline vs tuned on the held-out tasks — the headline comparison."""
    base = art.summary("base")
    tuned = art.summary("tuned")
    if not base and not tuned:
        return None

    metrics = [("cup", "CuP"), ("completion_rate", "Completion rate"),
               ("violation_rate", "Violation rate")]
    arms = [(name, blob, colour) for name, blob, colour in
            (("Baseline", base, theme.series[1]), ("Tuned", tuned, theme.series[0]))
            if blob]

    # Horizontal: three metrics across a wide canvas leaves each vertical slot
    # hundreds of pixels wide, so a 24px-capped column is lost in it and the two
    # arms' value labels land on top of each other. Rows give every bar its own
    # band of height, and the metric names read without rotation.
    n_arms = len(arms)
    fig, ax = plt.subplots(figsize=(7.0, 0.95 * len(metrics) + 2.0))

    ax.set_yticks(range(len(metrics)))
    ax.set_yticklabels([label for _, label in metrics])
    category_limits(ax, len(metrics), horizontal=True)
    ax.set_xlim(0, 1.16)
    # Limits before widths: fit_bar_width measures the axes to honour the 24px cap.
    bar_h = fit_bar_width(ax, 0.62 / max(n_arms, 1) * 0.86, horizontal=True)
    step = bar_h * 1.22                          # the leftover is the surface gap

    for a, (arm_name, blob, colour) in enumerate(arms):
        offset = (a - (n_arms - 1) / 2) * step
        for i, (key, _) in enumerate(metrics):
            value = float(blob.get(key) or 0.0)
            rounded_bar(ax, i + offset, value, bar_h, colour,
                        horizontal=True, zorder=3)
            ax.annotate(f"{value:.2f}", xy=(value, i + offset), xytext=(7, 0),
                        textcoords="offset points", ha="left", va="center",
                        fontsize=8, color=theme.text_secondary)
        ax.plot([], [], color=colour, linewidth=6, label=arm_name)

    ax.set_xlabel("Fraction of held-out episodes")
    n_eps = (tuned or base or {}).get("n_episodes", 0)
    titled(ax, "Completion under Policy — held-out tasks",
           f"{n_eps} episodes per arm · CuP counts only episodes that finish "
           f"AND break no policy", theme)
    if n_arms > 1:
        legend(ax, theme, loc="lower right")
    style_axes(ax, theme, xgrid=True)

    # An all-zero panel is honest but looks broken; say which it is.
    if all(not float(blob.get(k) or 0.0) for _, blob, _ in arms for k, _ in metrics):
        ax.annotate("every metric is zero — no episode completed a task",
                    xy=(0.5, 0.5), xycoords="axes fraction", ha="center",
                    fontsize=9.5, color=theme.text_muted)
    return fig


def fig_violations(art: RunArtifacts, theme: Theme) -> Optional[plt.Figure]:
    """Which safety dimensions actually fire. One series → one hue, no legend."""
    counts: dict[str, dict[str, int]] = {}
    for arm in ("base", "tuned"):
        summary = art.summary(arm) or {}
        for cat, n in (summary.get("violations_by_category") or {}).items():
            counts.setdefault(cat, {})[arm] = int(n)
    if not counts:
        return None

    cats = sorted(counts, key=lambda c: -sum(counts[c].values()))
    arms = [a for a in ("base", "tuned") if art.summary(a)]
    labels = {"base": "Baseline", "tuned": "Tuned"}
    colours = {"base": theme.series[1], "tuned": theme.series[0]}

    fig, ax = plt.subplots(figsize=(7.4, max(2.6, 0.62 * len(cats) + 1.8)))
    ax.set_yticks(range(len(cats)))
    ax.set_yticklabels([c.replace("_", " ") for c in cats])
    category_limits(ax, len(cats), horizontal=True)
    # The longest single bar, not the summed row — bars are grouped, not stacked.
    biggest = max(max(v.values()) for v in counts.values())
    ax.set_xlim(0, biggest * 1.16)
    bar_h = fit_bar_width(ax, 0.62 / max(len(arms), 1) * 0.86, horizontal=True)
    step = bar_h * 1.18

    for a, arm in enumerate(arms):
        offset = (a - (len(arms) - 1) / 2) * step
        for i, cat in enumerate(cats):
            value = counts[cat].get(arm, 0)
            if value:
                rounded_bar(ax, i + offset, value, bar_h, colours[arm],
                            horizontal=True, zorder=3)
                ax.annotate(str(value), xy=(value, i + offset), xytext=(6, 0),
                            textcoords="offset points", ha="left", va="center",
                            fontsize=8, color=theme.text_secondary)
        ax.plot([], [], color=colours[arm], linewidth=6, label=labels[arm])

    ax.set_xlabel("Violations")
    titled(ax, "Policy violations by dimension", "held-out episodes", theme)
    if len(arms) > 1:
        legend(ax, theme, loc="lower right")
    style_axes(ax, theme, xgrid=True)
    return fig


def fig_stage_timings(art: RunArtifacts, theme: Theme) -> Optional[plt.Figure]:
    """Where the wall-clock went — one series, emphasis on the slowest stage."""
    stages = (art.experiment or {}).get("stages") or {}
    rows = [(name, float(info.get("seconds") or 0.0))
            for name, info in stages.items() if info.get("seconds")]
    if not rows:
        return None

    names = [r[0] for r in rows]
    secs = [r[1] for r in rows]
    slowest = int(np.argmax(secs))

    fig, ax = plt.subplots(figsize=(7.4, max(2.4, 0.5 * len(rows) + 1.6)))
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names)
    category_limits(ax, len(names), horizontal=True)
    ax.set_xlim(0, max(secs) * 1.18)
    bar_h = fit_bar_width(ax, 0.5, horizontal=True)

    for i, (name, value) in enumerate(rows):
        # Emphasis, not categorical: one accent hue, the rest recede.
        colour = theme.series[0] if i == slowest else theme.de_emphasis
        rounded_bar(ax, i, value, bar_h, colour, horizontal=True, zorder=3)
        text = f"{value:.0f}s" if value < 90 else f"{value / 60.0:.1f} min"
        ax.annotate(text, xy=(value, i), xytext=(6, 0), textcoords="offset points",
                    ha="left", va="center", fontsize=8, color=theme.text_secondary)

    ax.set_xlabel("Seconds")
    titled(ax, "Stage wall-clock", "slowest stage highlighted", theme)
    style_axes(ax, theme, xgrid=True)
    return fig


FIGURES = [
    ("01_constraint_training", fig_constraint_training),
    ("02_gate_heldout", fig_gate),
    ("03_finetune_dynamics", fig_finetune),
    ("04_cup_comparison", fig_cup),
    ("05_violations_by_category", fig_violations),
    ("06_stage_timings", fig_stage_timings),
]


# ── HTML report ───────────────────────────────────────────────────────────────

def _tile(label: str, value: str, note: str = "") -> str:
    return (f'<div class="tile"><div class="tile-label">{html.escape(label)}</div>'
            f'<div class="tile-value">{html.escape(value)}</div>'
            f'<div class="tile-note">{html.escape(note)}</div></div>')


def _table(headers: list[str], rows: list[list[str]]) -> str:
    head = "".join(f"<th>{html.escape(h)}</th>" for h in headers)
    body = "".join(
        "<tr>" + "".join(f"<td>{html.escape(str(c))}</td>" for c in r) + "</tr>"
        for r in rows
    )
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, (int, float)):
        return f"{value:.{digits}f}" if isinstance(value, float) else str(value)
    return str(value)


def build_report(art: RunArtifacts, images: list[tuple[str, Path]],
                 theme: Theme) -> str:
    base = art.summary("base") or {}
    tuned = art.summary("tuned") or {}
    gate = art.gate or {}

    tiles = []
    if tuned or base:
        cup_b, cup_t = base.get("cup"), tuned.get("cup")
        if cup_t is not None and cup_b is not None:
            tiles.append(_tile("CuP (held out)", f"{cup_t:.3f}",
                               f"baseline {cup_b:.3f}  ({cup_t - cup_b:+.3f})"))
        elif cup_b is not None:
            tiles.append(_tile("CuP — baseline", f"{cup_b:.3f}", "no tuned arm yet"))
    if gate:
        tiles.append(_tile("Gate AUROC", _fmt(gate.get("auroc")),
                           ("passed" if gate.get("passed") else "FAILED")
                           + f" · threshold {_fmt(gate.get('auroc_gate'), 2)}"))
        tiles.append(_tile("Score separation", _fmt(gate.get("separation")),
                           "unsafe mean − safe mean"))
        tiles.append(_tile("Calibration (ECE)", _fmt(gate.get("ece")),
                           "lower is better"))

    figures_html = []
    for name, path in images:
        data = base64.b64encode(path.read_bytes()).decode()
        figures_html.append(
            f'<figure><img alt="{html.escape(name)}" '
            f'src="data:image/png;base64,{data}"/></figure>'
        )

    tables = []
    if gate.get("safe_scores") or gate.get("unsafe_scores"):
        rows = [[k, _fmt(gate.get(k))] for k in
                ("auroc", "f1", "ece", "separation", "safe_mean_score",
                 "unsafe_mean_score", "n_safe", "n_unsafe")]
        tables.append(("Constraint gate — held-out split",
                       _table(["metric", "value"], rows)))

    if base or tuned:
        keys = ["cup", "completion_rate", "violation_rate", "termination_rate",
                "mean_steps", "n_episodes", "n_errored_episodes"]
        rows = [[k, _fmt(base.get(k)), _fmt(tuned.get(k))] for k in keys]
        tables.append(("CuP evaluation", _table(["metric", "baseline", "tuned"], rows)))

    stages = (art.experiment or {}).get("stages") or {}
    if stages:
        rows = [[name, info.get("status", "—"),
                 _fmt(info.get("seconds"), 1) if info.get("seconds") else "—"]
                for name, info in stages.items()]
        tables.append(("Stages", _table(["stage", "status", "seconds"], rows)))

    tables_html = "".join(
        f"<section><h2>{html.escape(title)}</h2>{table}</section>"
        for title, table in tables
    )

    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>{html.escape(art.run_name)} — ICRL experiment report</title>
<style>
  :root {{
    color-scheme: {theme.name};
    --surface: {theme.surface};
    --ink: {theme.text_primary};
    --ink-2: {theme.text_secondary};
    --ink-3: {theme.text_muted};
    --rule: {theme.grid};
    --accent: {theme.series[0]};
  }}
  * {{ box-sizing: border-box; }}
  body {{
    margin: 0; padding: 40px 24px 72px; background: var(--surface); color: var(--ink);
    font: 15px/1.6 -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
  }}
  main {{ max-width: 980px; margin: 0 auto; }}
  h1 {{ font-size: 26px; margin: 0 0 4px; letter-spacing: -0.01em; }}
  .sub {{ color: var(--ink-3); font-size: 13.5px; margin-bottom: 32px; }}
  h2 {{ font-size: 15px; margin: 40px 0 12px; color: var(--ink-2);
        text-transform: uppercase; letter-spacing: 0.07em; font-weight: 600; }}
  .tiles {{ display: grid; gap: 12px; grid-template-columns: repeat(auto-fit, minmax(190px, 1fr)); }}
  .tile {{ border: 1px solid var(--rule); border-radius: 10px; padding: 16px 18px; }}
  .tile-label {{ font-size: 11.5px; text-transform: uppercase; letter-spacing: 0.06em;
                 color: var(--ink-3); }}
  .tile-value {{ font-size: 30px; font-weight: 600; letter-spacing: -0.02em; margin: 6px 0 2px; }}
  .tile-note {{ font-size: 12.5px; color: var(--ink-3); }}
  figure {{ margin: 0 0 28px; }}
  img {{ max-width: 100%; height: auto; display: block; }}
  table {{ border-collapse: collapse; width: 100%; font-size: 13.5px; }}
  th, td {{ text-align: left; padding: 7px 10px; border-bottom: 1px solid var(--rule); }}
  th {{ color: var(--ink-3); font-weight: 600; text-transform: uppercase;
        font-size: 11.5px; letter-spacing: 0.05em; }}
  td:not(:first-child), th:not(:first-child) {{ text-align: right;
        font-variant-numeric: tabular-nums; }}
  .wrap {{ overflow-x: auto; }}
  footer {{ margin-top: 48px; color: var(--ink-3); font-size: 12.5px; }}
</style></head>
<body><main>
  <h1>{html.escape(art.run_name)}</h1>
  <div class="sub">ICRL safety experiment · generated by scripts/make_experiment_plots.py</div>
  {'<div class="tiles">' + "".join(tiles) + '</div>' if tiles else ''}
  <h2>Figures</h2>
  {"".join(figures_html) or "<p>No figures — the run produced no plottable artifacts.</p>"}
  <div class="wrap">{tables_html}</div>
  <footer>Every number here is also in the JSON artifacts alongside this file.</footer>
</main></body></html>
"""


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--log-dir", type=Path, default=Path("logs"))
    ap.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints"))
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="default: <log-dir>/<run-name>/plots")
    ap.add_argument("--theme", choices=sorted(THEMES), default="light")
    ap.add_argument("--pdf", action="store_true", help="also write vector PDFs")
    ap.add_argument("--allow-empty", action="store_true",
                    help="exit 0 even when the run produced nothing to plot")
    args = ap.parse_args()

    art = RunArtifacts(args.run_name, args.log_dir, args.checkpoint_dir)
    if not art.anything:
        print(f"No artifacts for run '{args.run_name}' under {args.log_dir} / "
              f"{args.checkpoint_dir}.", file=sys.stderr)
        return 0 if args.allow_empty else 1

    theme = THEMES[args.theme]
    apply_theme(theme)

    out_dir = args.out_dir or (args.log_dir / args.run_name / "plots")
    out_dir.mkdir(parents=True, exist_ok=True)

    written: list[tuple[str, Path]] = []
    for name, builder in FIGURES:
        try:
            fig = builder(art, theme)
        except Exception as e:
            print(f"  skip {name}: {type(e).__name__}: {e}")
            continue
        if fig is None:
            print(f"  skip {name}: no data")
            continue
        png = out_dir / f"{name}.png"
        fig.savefig(png)
        if args.pdf:
            fig.savefig(out_dir / f"{name}.pdf")
        plt.close(fig)
        written.append((name, png))
        print(f"  wrote {png}")

    report = out_dir / "report.html"
    report.write_text(build_report(art, written, theme))
    print(f"  wrote {report}")
    print(f"\n{len(written)} figure(s) → {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
