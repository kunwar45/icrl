"""
Shared chart style: one place decides what every figure looks like.

Palette and mark specs follow a validated categorical scheme — the three
categorical slots clear the colourblind-separation, normal-vision and lightness
gates in both light and dark modes (all-pairs). Do not add a fourth series
colour: past three slots the all-pairs separation floors fail, which is why
every figure here caps at three series and folds the rest into a table.

Marks are deliberately thin — 2px lines, hairline solid gridlines, bars capped
in width with a rounded data-end — so the data is the only loud thing on the
page.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import matplotlib
matplotlib.use("Agg")  # no display on a compute node

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.path import Path as MplPath
from matplotlib.patches import PathPatch


@dataclass(frozen=True)
class Theme:
    name: str
    surface: str
    text_primary: str
    text_secondary: str
    text_muted: str
    grid: str
    series: tuple[str, ...]
    reference: str          # threshold / annotation hairlines
    de_emphasis: str        # "context" marks in an emphasis chart


LIGHT = Theme(
    name="light",
    surface="#fcfcfb",
    text_primary="#0b0b0b",
    text_secondary="#52514e",
    text_muted="#7c7a75",
    grid="#e5e4e0",
    series=("#2a78d6", "#eb6834", "#1baf7a"),
    reference="#9a9892",
    de_emphasis="#c9c7c1",
)

DARK = Theme(
    name="dark",
    surface="#1a1a19",
    text_primary="#ffffff",
    text_secondary="#c3c2b7",
    text_muted="#8f8e85",
    grid="#33332f",
    series=("#3987e5", "#d95926", "#199e70"),
    reference="#6f6e68",
    de_emphasis="#4a4a46",
)

THEMES = {"light": LIGHT, "dark": DARK}


def apply_theme(theme: Theme) -> None:
    """Set rcParams so individual figures don't restate styling."""
    plt.rcParams.update({
        "figure.facecolor": theme.surface,
        "savefig.facecolor": theme.surface,
        "axes.facecolor": theme.surface,
        "axes.edgecolor": theme.grid,
        "axes.labelcolor": theme.text_secondary,
        "axes.titlecolor": theme.text_primary,
        "axes.linewidth": 1.0,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": theme.grid,
        "grid.linewidth": 1.0,
        "grid.linestyle": "-",          # never dashed
        "xtick.color": theme.text_secondary,
        "ytick.color": theme.text_secondary,
        "xtick.labelcolor": theme.text_secondary,
        "ytick.labelcolor": theme.text_secondary,
        "text.color": theme.text_primary,
        "font.size": 9,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",   # "600" isn't a real weight in most fonts
        "axes.titlepad": 10,
        "legend.frameon": False,
        "legend.fontsize": 9,
        "lines.linewidth": 2.0,         # 2px lines
        "lines.solid_capstyle": "round",
        "lines.solid_joinstyle": "round",
        "figure.dpi": 150,
        "savefig.dpi": 150,
        "savefig.bbox": "tight",
    })


def style_axes(ax, theme: Theme, *, xgrid: bool = False) -> None:
    """Recessive chrome: no box, hairline grid on one axis only."""
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(theme.grid)
    ax.grid(axis="x" if xgrid else "y", color=theme.grid, linewidth=1.0)
    ax.grid(axis="y" if xgrid else "x", visible=False)
    ax.tick_params(length=0)


# ── Marks ─────────────────────────────────────────────────────────────────────

def fit_bar_width(ax, desired: float, *, horizontal: bool = False,
                  max_px: float = 24.0) -> float:
    """
    Clamp a bar's thickness to the 24px cap, in data units.

    Hand-tuning a fraction-of-slot breaks the moment the figure is resized: the
    same 0.5 slot is a 20px bar in one chart and a 90px slab in another. Measure
    the axes in pixels and convert the cap back into data units.

    Call after the relevant axis limits are final.
    """
    fig = ax.figure
    fig.canvas.draw_idle()
    try:
        bbox = ax.get_window_extent(renderer=fig.canvas.get_renderer())
    except Exception:
        return desired
    span_px = bbox.height if horizontal else bbox.width
    lo, hi = ax.get_ylim() if horizontal else ax.get_xlim()
    data_span = abs(hi - lo) or 1.0
    px_per_unit = span_px / data_span
    if px_per_unit <= 0:
        return desired
    return min(desired, max_px / px_per_unit)


def category_limits(ax, n: int, *, horizontal: bool = False,
                    pad: float = 0.62) -> None:
    """Pad a categorical axis so the first and last bars aren't clipped."""
    if horizontal:
        ax.set_ylim(n - 1 + pad, -pad)   # inverted: first category on top
    else:
        ax.set_xlim(-pad, n - 1 + pad)


def rounded_bar(ax, x, y, width, color, *, horizontal: bool = False,
                radius_frac: float = 0.28, baseline: float = 0.0, **kw):
    """
    A bar with a rounded data-end and a square baseline.

    matplotlib's Rectangle rounds all corners or none; the spec wants the growth
    end rounded and the baseline flush, which reads as "grows from the axis"
    rather than "floating pill".
    """
    length = y - baseline
    if length == 0:
        return None
    r = min(abs(length), width) * radius_frac
    sign = 1 if length >= 0 else -1
    r *= sign

    if horizontal:
        x0, x1 = baseline, y
        y0, y1 = x - width / 2, x + width / 2
        verts = [
            (x0, y0), (x1 - r, y0), (x1, y0), (x1, y0 + abs(r)),
            (x1, y1 - abs(r)), (x1, y1), (x1 - r, y1), (x0, y1), (x0, y0),
        ]
        codes = [MplPath.MOVETO, MplPath.LINETO, MplPath.CURVE3, MplPath.CURVE3,
                 MplPath.LINETO, MplPath.CURVE3, MplPath.CURVE3, MplPath.LINETO,
                 MplPath.CLOSEPOLY]
    else:
        x0, x1 = x - width / 2, x + width / 2
        y0, y1 = baseline, y
        verts = [
            (x0, y0), (x0, y1 - r), (x0, y1), (x0 + width * radius_frac, y1),
            (x1 - width * radius_frac, y1), (x1, y1), (x1, y1 - r), (x1, y0),
            (x0, y0),
        ]
        codes = [MplPath.MOVETO, MplPath.LINETO, MplPath.CURVE3, MplPath.CURVE3,
                 MplPath.LINETO, MplPath.CURVE3, MplPath.CURVE3, MplPath.LINETO,
                 MplPath.CLOSEPOLY]

    patch = PathPatch(MplPath(verts, codes), facecolor=color, edgecolor="none", **kw)
    ax.add_patch(patch)
    return patch


def end_marker(ax, x, y, color, theme: Theme, size: float = 8.0):
    """>=8px marker carrying a 2px surface ring so it stays legible on crossings."""
    ax.plot([x], [y], marker="o", markersize=size, color=color,
            markeredgecolor=theme.surface, markeredgewidth=2.0, zorder=5,
            linestyle="none")


def reference_line(ax, value: float, label: str, theme: Theme, *,
                   horizontal: bool = True, x_frac: float = 0.012):
    """
    A threshold annotation — chrome, never a data series.

    Labelled on the LEFT by default: series end-markers and their direct labels
    live at the right edge, and a right-aligned threshold label lands on top of
    them.
    """
    if horizontal:
        ax.axhline(value, color=theme.reference, linewidth=1.0, zorder=1)
        ax.annotate(label, xy=(x_frac, value), xycoords=("axes fraction", "data"),
                    xytext=(0, 4), textcoords="offset points",
                    ha="left", va="bottom", fontsize=8, color=theme.text_muted)
    else:
        ax.axvline(value, color=theme.reference, linewidth=1.0, zorder=1)
        ax.annotate(label, xy=(value, 0.98), xycoords=("data", "axes fraction"),
                    ha="left", va="top", fontsize=8, color=theme.text_muted)


def integer_axis(ax, which: str = "x") -> None:
    """Iterations and steps are whole numbers — never label one '2.5'."""
    from matplotlib.ticker import MaxNLocator
    axis = ax.xaxis if which == "x" else ax.yaxis
    axis.set_major_locator(MaxNLocator(integer=True, nbins="auto"))


def label_room(ax, frac: float = 0.12) -> None:
    """Widen the right limit so end-of-line labels are not clipped by the axes."""
    lo, hi = ax.get_xlim()
    ax.set_xlim(lo, hi + (hi - lo) * frac)


def label_endpoints(ax, entries: Sequence[tuple[float, float, str]], theme: Theme,
                    *, dx: float = 9.0, min_gap_frac: float = 0.07) -> None:
    """
    Direct-label series endpoints, pushing labels apart when they'd overlap.

    Two series converging to nearly the same value is the normal case here
    (expert and policy scores, base and tuned rates), and stacked text at the
    same y is unreadable. Call after the y-limits are final.

    entries: (x, y, text) — y is the true data value; only the label moves.
    """
    if not entries:
        return
    lo, hi = ax.get_ylim()
    min_gap = (hi - lo) * min_gap_frac

    items = sorted(entries, key=lambda e: e[1])
    ys = [e[1] for e in items]
    for i in range(1, len(ys)):
        if ys[i] - ys[i - 1] < min_gap:
            ys[i] = ys[i - 1] + min_gap
    # Keep the whole stack inside the axes if pushing overflowed the top.
    overflow = ys[-1] - (hi - min_gap * 0.5)
    if overflow > 0:
        ys = [y - overflow for y in ys]

    for (x, _, text), y in zip(items, ys):
        ax.annotate(text, xy=(x, y), xytext=(dx, 0), textcoords="offset points",
                    ha="left", va="center", fontsize=8, color=theme.text_secondary,
                    zorder=6)


def direct_label(ax, x, y, text: str, color: str, theme: Theme, *,
                 dx: float = 0.0, dy: float = 0.0, ha: str = "left",
                 va: str = "center"):
    """
    Label text never wears the series colour — identity comes from the mark
    beside it. `color` is accepted for call-site symmetry and ignored on purpose.
    """
    ax.annotate(text, xy=(x, y), xytext=(dx, dy), textcoords="offset points",
                ha=ha, va=va, fontsize=8, color=theme.text_secondary, zorder=6)


def titled(ax, title: str, subtitle: str = "", theme: Theme = LIGHT) -> None:
    """
    Title with an optional subtitle beneath it.

    Both are placed in *points* above the axes rather than axes fractions, so
    the gap is identical whether the panel is 2in or 6in tall — placing the
    subtitle at a fixed fraction puts it on top of the title in short panels.
    """
    ax.set_title(title, loc="left", color=theme.text_primary,
                 pad=24 if subtitle else 10)
    if subtitle:
        ax.annotate(subtitle, xy=(0, 1.0), xycoords="axes fraction",
                    xytext=(0, 6), textcoords="offset points",
                    ha="left", va="bottom", fontsize=8.5, color=theme.text_muted)


def legend(ax, theme: Theme, **kw) -> None:
    """A legend is always present for two or more series."""
    leg = ax.legend(loc=kw.pop("loc", "best"), **kw)
    if leg:
        for text in leg.get_texts():
            text.set_color(theme.text_secondary)


def empty_note(ax, message: str, theme: Theme) -> None:
    """A panel with nothing to draw says so, rather than rendering blank axes."""
    ax.axis("off")
    ax.text(0.5, 0.5, message, ha="center", va="center", fontsize=9,
            color=theme.text_muted, transform=ax.transAxes, wrap=True)
