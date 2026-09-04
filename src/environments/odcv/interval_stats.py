# ABOUTME: Clustered confidence intervals vendored from the LASR repo (src/eval/stats.py): scenario-as-item design, log-odds bounds, Satterthwaite pooling
# ABOUTME: Import: from src.environments.odcv.interval_stats import Design, Result, interval; vendored 2026-09-04 from lessons_from_constitutional_aft @ 5232eca
# ABOUTME: Eval-agnostic error bars: closed-form intervals for a checkpoints x items table under
# ABOUTME: declared Design, paired differences, and a cluster bootstrap for non-mean statistics.

"""Error bars for any eval that produces a table of per-cell outcomes.

The picture (derived in docs/error_bars.md; Miller, arXiv:2411.00640, plus a model axis):
every cell score is a true rate plus rollout noise; the true rate splits into a checkpoint level,
an item level and their interaction; the four pieces are uncorrelated so variances add, each over
the number of independent draws of it:

    Var(mu_hat) = s_A^2/n + s_B^2/J + s_C^2/(nJ) + s_eps^2/(nJR)

None of the four is observable. Three spreads of the table are, and combine to it exactly:

    T_A  spread of the n row means / n          -> s_A^2/n + beta        (Miller's clustered SE, by model)
    T_B  spread of the J column means / J       -> s_B^2/J + beta        (Miller's clustered SE, by unit)
    T_C  spread of the double-centred residuals -> beta                  (interaction + rollout noise)
    Var(mu_hat) is estimated by T_A + T_B - T_C,  CI = mu_hat +/- t_nu SE.

The multiplier is a t quantile, never a flat 1.96: the variance is ESTIMATED, and a noisy
estimate needs a fatter multiplier to keep 95% coverage. df is how many independent numbers
went into it -- J-1 for a per-item spread, n-1 for a per-checkpoint one. It matters at small
counts: at df 39, +/-1.96 covers 94.3%; at df 2 (three seeds) it covers 81%. Sums of
estimates with different dfs (T_A + T_B - T_C) have no exact df, so `satterthwaite` gives an
effective one -- roughly the df of whichever part dominates the sum.

A `Design` names the factors of the long table and what each one is:

    item           the sampled draw from the benchmark population (scenario, question, prompt)
    enumerated     factors whose levels are ALL present in every item, with fixed weights (ODCV's
                   two variants at 1/2 each; Arena-Hard's two orderings). Enumerated, not
                   sampled: collapsed into the cell, no variance term, they only fix the estimand.
    subsamples     draws inside a cell with no identity across cells (rollouts, questions within
                   a subject, judge calls). Averaged in; live only inside beta.
    item_sampling  "sampled" (generalise to the population) or "fixed" (this benchmark).

The checkpoint axis is NOT declared: it is inferred at call time -- `checkpoints="sampled"` needs
n >= 2 of them from one pipeline and adds T_A and T_C; n == 1 is `checkpoints="fixed"`, a claim
about that one checkpoint. A config cannot assert a sample size it does not have.

Worked Designs. There is always exactly one `item`; what sits inside it is `subsamples`, what
sits beside it with named levels is `enumerated`:

    ODCV, the 50/50 mixture      Design(item="scenario", enumerated={"variant": "equal"},
                                        subsamples=("pass",))
    ODCV, one variant            Design(item="scenario", subsamples=("pass",))
    MMLU, Miller's framing       Design(item="question")
        -- a question drawn from the MMLU-like question population; SE = sqrt(p(1-p)/n).
    MMLU, stratified by subject  Design(item="subject", item_sampling="fixed",
                                        item_weights="count", subsamples=("question",))
        -- MMLU's own subject mix, between-subject variance removed; narrower.
    Arena-Hard                   Design(item="prompt", enumerated={"ordering": "equal"},
                                        subsamples=("judge_call",))

The two MMLU rows are different claims, not different spellings of one: pick deliberately.

Rollouts. With R > 1 the within-cell spread estimates s_eps^2, so the rollout share of the error
bar is reported. With R == 1 the interval is still valid -- rollout noise sits inside every
spread and is measured with it -- but it cannot be separated, and the cell value has to be read
as the checkpoint's behaviour on that item. The one thing R == 1 cannot support is the
both-fixed question ("these checkpoints on these items", where rollouts are the only randomness):
raises `NotEstimable` instead of returning a zero-width bar.

Nothing here is bootstrapped except `cluster_bootstrap`, for statistics with no closed form
(Bradley-Terry ratings, medians). For a mean it agrees with `interval` to Monte Carlo error.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict
from statistics import NormalDist
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd

Z_95 = 1.959963984540054
__all__ = [
    "Design", "Table", "Result", "NotEstimable", "collapse", "spreads", "satterthwaite",
    "interval", "difference", "cluster_bootstrap", "wilson", "mcnemar_exact", "t_quantile",
]


class NotEstimable(ValueError):
    """The requested interval cannot be computed from this data, and why."""


# --------------------------------------------------------------------------- design + table

@dataclass(frozen=True)
class Design:
    """What each column of the long table is. See the module docstring.

    Attributes:
        item: Column naming the benchmark item -- the sampled draw (scenario, question, prompt).
        item_sampling: "sampled" (items are a draw from a population you generalise to) or
            "fixed" (the items ARE the benchmark; no item-to-item term).
        enumerated: `{column: "equal" | {level: weight}}` -- enumerated factors, all levels
            required in every cell, collapsed with these weights.
        subsamples: Columns of within-cell draws (rollout/pass, question, judge call), averaged in.

        value: Column holding the outcome (0/1 or a score).
        item_weights: "equal" or "count" (weight a fixed item by its number of observations,
            e.g. MMLU's question-count weighting). Only valid with item_sampling='fixed'.
        incomplete: "drop" items missing a checkpoint or an enumerated level (recorded in
            `Table.dropped_items`), or "error".
    """
    item: str
    item_sampling: str = "sampled"
    enumerated: Mapping[str, Any] = field(default_factory=dict)
    subsamples: tuple[str, ...] = ()
    checkpoint: str = "checkpoint"
    value: str = "value"
    item_weights: str = "equal"
    incomplete: str = "drop"

    def __post_init__(self):
        assert self.item_sampling in ("sampled", "fixed"), \
            f"item_sampling must be sampled|fixed, got {self.item_sampling!r}"
        assert self.item_weights in ("equal", "count"), f"item_weights must be equal|count"
        assert self.incomplete in ("drop", "error"), f"incomplete must be drop|error"
        assert not (self.item_weights == "count" and self.item_sampling == "sampled"), (
            "item_weights='count' with item_sampling='sampled' is incoherent, so it is refused rather "
            "than silently ignored: weighting a SAMPLED item by however many observations it "
            "happened to receive implies the population you are generalising to is over those "
            "observations, not over the items -- in which case make the observation the item. "
            "Count-weighting is for fixed strata (MMLU's 57 subjects by question count).")
        object.__setattr__(self, "subsamples", tuple(self.subsamples))
        object.__setattr__(self, "enumerated", dict(self.enumerated))
        for col, spec in self.enumerated.items():
            assert spec == "equal" or (isinstance(spec, Mapping) and spec), \
                f"enumerated[{col!r}] must be 'equal' or a {{level: weight}} mapping"

    @classmethod
    def from_config(cls, cfg: Mapping[str, Any]) -> "Design":
        """Build from a config block (dict / OmegaConf); unknown keys are an error."""
        d = {k: (dict(v) if isinstance(v, Mapping) else (list(v) if isinstance(v, (list, tuple)) else v))
             for k, v in dict(cfg).items()}
        allowed = {f for f in cls.__dataclass_fields__}
        unknown = set(d) - allowed
        assert not unknown, f"unknown Design keys {sorted(unknown)}; allowed: {sorted(allowed)}"
        if "subsamples" in d:
            d["subsamples"] = tuple(d["subsamples"])
        return cls(**d)

    def fixed_weights(self, col: str, levels: Sequence[Any]) -> dict[Any, float]:
        spec = self.enumerated[col]
        if spec == "equal":
            return {lv: 1.0 / len(levels) for lv in levels}
        missing = set(levels) - set(spec)
        assert not missing, f"enumerated[{col!r}] has no weight for levels {sorted(map(str, missing))}"
        total = float(sum(spec[lv] for lv in levels))
        return {lv: float(spec[lv]) / total for lv in levels}

    def describe_items(self) -> str:
        return ("items sampled from the benchmark population" if self.item_sampling == "sampled"
                else "these items as a fixed benchmark")


@dataclass
class Table:
    """The n x J table of cell scores a Design collapses a long table to.

    Attributes:
        values: (n, J) cell means (over fixed levels with their weights, then subsamples draws).
        checkpoints, items: Row and column labels.
        counts: (n, J) number of subsamples observations behind each cell, all levels together.
        reps: (n, J) rollouts per fixed level in the cell (the smallest level's count) --
            the R of the derivation; `counts` is R times the number of fixed-level combos.
        within_cell_var: (n, J) variance of the cell mean due to subsamples draws, estimated from the
            within-cell spread: sum over fixed levels of w^2 s^2 / R. NaN where any level has
            fewer than two draws.
        item_weights: (J,) weights summing to 1 (equal unless the Design says otherwise).
        dropped_items: Items removed for being absent, or incomplete, under some checkpoint.
        design: The Design that produced it.
    """
    values: np.ndarray
    checkpoints: list[str]
    items: list[str]
    counts: np.ndarray
    reps: np.ndarray
    within_cell_var: np.ndarray
    item_weights: np.ndarray
    dropped_items: list[str]
    design: Design

    @property
    def n_checkpoints(self) -> int:
        return len(self.checkpoints)

    @property
    def n_items(self) -> int:
        return len(self.items)

    def rollouts(self) -> dict[str, float]:
        return {"min": int(self.reps.min()), "max": int(self.reps.max()),
                "mean": float(self.reps.mean())}

    def select_items(self, items: Sequence[str]) -> "Table":
        idx = [self.items.index(u) for u in items]
        w = self.item_weights[idx]
        return Table(self.values[:, idx], list(self.checkpoints), list(items), self.counts[:, idx],
                     self.reps[:, idx], self.within_cell_var[:, idx], w / w.sum(),
                     list(self.dropped_items), self.design)


def _frame(obs: Any) -> pd.DataFrame:
    if isinstance(obs, pd.DataFrame):
        return obs.copy()
    return pd.DataFrame(list(obs))


def collapse(obs: Any, design: Design) -> Table:
    """Long table -> n x J table of cell scores, honouring the Design's collapse rules.

    Args:
        obs: Rows with at least the columns the Design names (DataFrame or iterable of dicts).
        design: The Design.

    Returns:
        A balanced Table: every kept item is present and complete for every checkpoint.
    """
    df = _frame(obs)
    fixed_cols = list(design.enumerated)
    need = [design.checkpoint, design.item, design.value, *fixed_cols]
    missing = [c for c in need if c not in df.columns]
    assert not missing, f"long table lacks columns {missing}; has {list(df.columns)}"
    stray = [c for c in design.subsamples if c not in df.columns]
    assert not stray, f"subsamples columns {stray} not in the long table"
    assert len(df), "empty long table"
    df = df[[design.checkpoint, design.item, design.value, *fixed_cols]].copy()
    df[design.value] = df[design.value].astype(float)

    keys = [design.checkpoint, design.item, *fixed_cols]
    g = df.groupby(keys, sort=True)[design.value]
    per_level = pd.DataFrame({"mean": g.mean(), "count": g.count(), "var": g.var(ddof=1)})
    checkpoints = sorted(df[design.checkpoint].unique(), key=str)
    items = sorted(df[design.item].unique(), key=str)
    fixed_levels = {c: sorted(df[c].unique(), key=str) for c in fixed_cols}
    weights = {c: design.fixed_weights(c, fixed_levels[c]) for c in fixed_cols}
    combos = [()] if not fixed_cols else list(_product([fixed_levels[c] for c in fixed_cols]))
    combo_w = {combo: float(np.prod([weights[c][lv] for c, lv in zip(fixed_cols, combo)])) for combo in combos}

    idx = per_level.index
    values = np.full((len(checkpoints), len(items)), np.nan)
    counts = np.zeros((len(checkpoints), len(items)), dtype=int)
    reps = np.zeros((len(checkpoints), len(items)), dtype=int)
    noise = np.full((len(checkpoints), len(items)), np.nan)
    complete = np.ones((len(checkpoints), len(items)), dtype=bool)
    for i, m in enumerate(checkpoints):
        for j, u in enumerate(items):
            v = c = nv = 0.0
            r_min = None
            for combo in combos:
                key = (m, u, *combo) if fixed_cols else (m, u)
                if key not in idx:
                    complete[i, j] = False
                    break
                row = per_level.loc[key]
                w, cnt = combo_w[combo], int(row["count"])
                v += w * float(row["mean"])
                c += cnt
                r_min = cnt if r_min is None else min(r_min, cnt)
                nv += w * w * (float(row["var"]) / cnt if cnt >= 2 else np.nan)
            if complete[i, j]:
                values[i, j], counts[i, j], reps[i, j], noise[i, j] = v, int(c), int(r_min), nv

    keep = complete.all(axis=0)
    dropped = [u for u, k in zip(items, keep) if not k]
    if dropped and design.incomplete == "error":
        raise NotEstimable(f"{len(dropped)} item(s) missing a checkpoint or an enumerated level: {dropped[:10]}"
                           + (" ..." if len(dropped) > 10 else ""))
    kept = [j for j, k in enumerate(keep) if k]
    assert len(kept) >= 2, f"fewer than two complete items (dropped {dropped})"
    values, counts, reps, noise = values[:, kept], counts[:, kept], reps[:, kept], noise[:, kept]
    items = [items[j] for j in kept]
    if design.item_sampling == "fixed" and design.item_weights == "count":
        uw = counts.sum(axis=0).astype(float)
    else:
        uw = np.ones(len(items))
    return Table(values, [str(m) for m in checkpoints], [str(u) for u in items], counts, reps, noise,
                 uw / uw.sum(), [str(u) for u in dropped], design)


def _product(levels: list[list[Any]]):
    if not levels:
        yield ()
        return
    for first in levels[0]:
        for rest in _product(levels[1:]):
            yield (first, *rest)


# --------------------------------------------------------------------------- the three spreads

def satterthwaite(parts: Sequence[tuple[float, float]]) -> float:
    """Effective df for a variance built as a sum of estimated parts.

    Each part is `(value, df)`. A t-interval assumes ONE variance estimate with a known df;
    `T_A + T_B - T_C` mixes estimates whose dfs differ (n-1, J-1, (n-1)(J-1)) and has no
    exact df. Satterthwaite matches the first two moments of the sum to a single scaled
    chi-square, giving

        nu = (sum of parts)^2 / sum over parts of (part^2 / df_part)

    which behaves as it should at the ends: if one part dominates, nu is that part's df; if
    several contribute equally, nu is larger than any single one, because averaging several
    noisy variance estimates gives a less noisy total. Signs are irrelevant in the
    denominator, so a subtracted term still costs df rather than adding it.

    Args:
        parts: `(value, df)` for each estimated component; parts with df <= 0 or value 0
            contribute nothing.

    Returns:
        The effective degrees of freedom (>= 1), or inf when nothing is estimated.
    """
    total = sum(v for v, _ in parts)
    denom = sum(v * v / d for v, d in parts if d and d > 0)
    if denom <= 0 or total <= 0:
        return float("inf")
    return max(1.0, total * total / denom)


def spreads(values: np.ndarray, item_weights: np.ndarray | None = None) -> dict[str, float]:
    """mu_hat and T_A, T_B, T_C for an n x J table (n >= 2 for T_A and T_C; NaN otherwise)."""
    v = np.asarray(values, dtype=float)
    assert v.ndim == 2 and v.shape[1] >= 2, f"need an n x J table with J >= 2, got {v.shape}"
    n_checkpoints, n_items = v.shape
    w = (np.ones(n_items) / n_items if item_weights is None
         else np.asarray(item_weights, float))
    # (v * w).sum rather than `v @ w`: numpy's BLAS matmul path raises spurious
    # divide-by-zero/overflow RuntimeWarnings on some shapes (reproducible with a plain
    # `@` on the same arrays, numpy 2.2), which would pollute stderr for every caller.
    row = (v * w[None, :]).sum(axis=1)   # per-checkpoint rates (item-weighted)
    col = v.mean(axis=0)              # per-item rates
    mu = float(row.mean())
    out = {"mu": mu, "T_A": float("nan"), "T_B": float(col.var(ddof=1) / n_items),
           "T_C": float("nan")}
    if n_checkpoints >= 2:
        resid = v - row[:, None] - col[None, :] + mu
        out["T_A"] = float(row.var(ddof=1) / n_checkpoints)
        out["T_C"] = float((resid ** 2).sum() / ((n_checkpoints - 1) * (n_items - 1))
                           / (n_checkpoints * n_items))
    return out


# --------------------------------------------------------------------------- results

@dataclass
class Result:
    """One interval and everything needed to read it honestly."""
    estimand: str
    method: str
    mean: float
    se: float
    lo: float
    hi: float
    mult: float
    df: float
    n_checkpoints: int
    n_items: int
    checkpoint_sampling: str
    item_sampling: str
    terms: dict[str, float] = field(default_factory=dict)
    rollouts: dict[str, float] = field(default_factory=dict)
    noise: dict[str, Any] = field(default_factory=dict)
    claims: list[str] = field(default_factory=list)
    dropped_items: list[str] = field(default_factory=list)
    shape: str = "symmetric"          # "symmetric" | "logit" | "wilson-at-boundary"
    lo_symmetric: float = 0.0         # mean -/+ mult*se, kept on the record even when the
    hi_symmetric: float = 0.0         # reported interval is the log-odds one

    def as_dict(self) -> dict:
        d = asdict(self)
        d["terms"] = {k: _jsonable(v) for k, v in self.terms.items()}
        d["noise"] = {k: _jsonable(v) for k, v in self.noise.items()}
        d["ci95"] = [self.lo, self.hi]
        d["ci95_symmetric"] = [self.lo_symmetric, self.hi_symmetric]
        return d


def _jsonable(v):
    if isinstance(v, np.generic):
        return v.item()
    if isinstance(v, float) and not math.isfinite(v):
        return None
    return v


def _within_cell_parts(table: Table) -> list[tuple[float, float]]:
    """Each cell's contribution to Var(mu_hat) when rollouts are the only randomness.

    `(w_j^2 * noise_var_ij / n^2, R_ij - 1)` per cell, ready for `satterthwaite`. Pooling
    these as one estimate with df = sum(R-1) is exact only when every cell has the same R
    AND the same noise; Satterthwaite reduces to that when they do (K equal parts of df d
    give nu = K*d) and correctly loses df when one noisy cell dominates.
    """
    w = table.item_weights
    return [(float(w[j] ** 2 * table.within_cell_var[i, j] / table.n_checkpoints ** 2), float(table.reps[i, j] - 1))
            for i in range(table.n_checkpoints) for j in range(table.n_items)]


def _within_cell_block(table: Table) -> dict[str, Any]:
    """Rollout-noise contribution to Var(mu_hat), where the data can support it."""
    nv = table.within_cell_var
    estimable = bool(np.isfinite(nv).all())
    if not estimable:
        return {"estimable": False, "sigma_eps2": None, "term": None, "share": None,
                "reason": "every cell needs >= 2 draws of every level to estimate rollout noise"}
    w = table.item_weights
    term = float(((nv * w[None, :] ** 2).sum(axis=1)).sum() / table.n_checkpoints ** 2)
    per_draw = float(np.nanmean(nv * table.counts))   # s^2 per draw, roughly
    return {"estimable": True, "sigma_eps2": per_draw, "term": term, "share": None}


def _claims(table: Table, checkpoint_sampling: str) -> list[str]:
    d = table.design
    out = []
    out.append(f"{'models sampled' if checkpoint_sampling == "sampled" else 'model(s) fixed'}: "
               + ("generalises to checkpoints from the same training pipeline "
                  f"(n={table.n_checkpoints} seeds; seed-to-seed variance estimated)" if checkpoint_sampling == "sampled"
                  else f"about {'this checkpoint' if table.n_checkpoints == 1 else f'these {table.n_checkpoints} checkpoints'} only; "
                       "pipeline (seed-to-seed) variance is not estimated"))
    out.append(("items sampled: generalises to items drawn like these "
                f"({table.n_items} {d.item}s; item-to-item variance estimated)") if d.item_sampling == "sampled"
               else f"units fixed: about these {table.n_items} {d.item}s only; no item-to-item term")
    r = table.rollouts()
    if r["max"] == 1:
        out.append("one rollout per cell: rollout noise is inside every spread and is measured "
                   "with it, but cannot be separated; a cell's value is read as the checkpoint's "
                   f"behaviour on that {d.item}")
    else:
        out.append(f"{r['min']}-{r['max']} draws per cell: rollout noise estimated from "
                   "within-cell spread (see `noise`)")
    if d.enumerated:
        out.append("fixed factors " + ", ".join(f"{c} ({'equal' if s == 'equal' else 'weighted'})"
                                                 for c, s in d.enumerated.items())
                   + " are enumerated in every cell: in the estimand, no variance term")
    if table.dropped_items:
        out.append(f"{len(table.dropped_items)} incomplete item(s) dropped to keep the table balanced")
    return out


def _estimand(table: Table, checkpoint_sampling: str) -> str:
    d = table.design
    who = ("a checkpoint from the pipeline" if checkpoint_sampling == "sampled"
           else ("this checkpoint" if table.n_checkpoints == 1 else f"the mean of these {table.n_checkpoints} checkpoints"))
    where = (f"a {d.item} drawn like these" if d.item_sampling == "sampled" else f"these {table.n_items} {d.item}s")
    mix = ("" if not d.enumerated else
           " under the fixed mix of " + " x ".join(d.enumerated))
    return f"mean outcome of {who} on {where}{mix}"


def _boundary_n(table: Table, checkpoint_sampling: str) -> int:
    """Draws on the smallest SAMPLED axis -- the axis that limits what the data can rule out.

    Only used when the estimate sits on the edge of its range, where there is no spread to
    scale. With 3 checkpoints and 40 items all reading zero we have 40 draws bounding the item
    distribution but only 3 bounding the checkpoint distribution, so 3 is the honest count: an
    unobserved checkpoint could still misbehave, and no number of items rules that out.
    """
    axes = []
    if checkpoint_sampling == "sampled":
        axes.append(table.n_checkpoints)
    if table.design.item_sampling == "sampled":
        axes.append(table.n_items)
    if axes:
        return max(min(axes), 1)
    return max(int(np.nansum(table.reps)), 1)   # both fixed: rollouts are the only randomness


def _shape_interval(mean: float, se: float, mult: float, bounds: tuple[float, float] | None,
                    n_floor: int, z: float = Z_95) -> tuple[float, float, str]:
    """`lo`, `hi` and the shape used. Symmetric unless the value is declared a bounded rate.

    `mean +/- t*SE` assumes the sampling distribution of the mean is symmetric. For a rate near
    its floor it is not: the mean cannot go below the floor but can go well above, so the
    distribution is right-skewed, the symmetric interval sits too far left and under-covers, and
    its lower end can fall outside the range entirely. Taking the same `+/- t*SE` step on the
    log-odds scale -- where the boundary is at infinity -- and mapping back gives an asymmetric
    interval that cannot leave the range. Delta method: SD(logit p) = SE / (p(1-p)).

    The estimand and the SE are untouched; only the geometry of the interval around them changes.
    """
    if bounds is None:
        return mean - mult * se, mean + mult * se, "symmetric"
    lo_b, hi_b = bounds
    span = hi_b - lo_b
    p, s = (mean - lo_b) / span, se / span
    if s > 0 and 0.0 < p < 1.0:
        half = mult * s / (p * (1.0 - p))
        centre = math.log(p / (1.0 - p))
        return (lo_b + span / (1.0 + math.exp(-(centre - half))),
                lo_b + span / (1.0 + math.exp(-(centre + half))), "logit")
    # On the edge of the range (or every cell identical): no log-odds, and every spread is zero,
    # so the symmetric interval would be a POINT -- certainty from a finite sample. Fall back to
    # the binomial score interval at the binding axis to get a real bound. It takes the NORMAL z,
    # not `mult`: Wilson inverts a score test whose nominal level is defined by z, and `mult`
    # here is a t on a df estimated from spreads that are all zero (typically inf, but a
    # near-degenerate table can make it 2, which would move this bound for no honest reason).
    lo, hi = wilson(p * n_floor, n_floor, z=z)
    return lo_b + span * lo, lo_b + span * hi, "wilson-at-boundary"


def _finish(table: Table, checkpoint_sampling: str, method: str, mean: float, se2: float, df: float,
            terms: dict[str, float], alpha: float,
            bounds: tuple[float, float] | None = None) -> Result:
    se = math.sqrt(max(se2, 0.0))
    mult = t_quantile(1 - alpha / 2, df)
    noise = _within_cell_block(table)
    if noise["estimable"] and se2 > 0:
        noise["share"] = float(noise["term"] / se2)
    n_floor = _boundary_n(table, checkpoint_sampling)
    lo, hi, shape = _shape_interval(float(mean), se, mult, bounds, n_floor,
                                    z=t_quantile(1 - alpha / 2, math.inf))
    claims = _claims(table, checkpoint_sampling)
    if shape == "logit":
        claims.append(f"the value is a rate on [{bounds[0]:g}, {bounds[1]:g}], so the interval is "
                      "built on the log-odds scale and is asymmetric; mean and SE are unchanged")
    elif shape == "wilson-at-boundary":
        claims.append(f"DEGENERATE: every spread estimate is 0, so the `terms`, `se` and `df` "
                      f"above measure nothing and the claims above them are vacuous. The estimate "
                      f"sits on the edge of [{bounds[0]:g}, {bounds[1]:g}], so the reported "
                      f"interval is instead a binomial score bound at n={n_floor} -- the draws on "
                      f"the smallest sampled axis -- i.e. what that many draws cannot rule out")
    return Result(_estimand(table, checkpoint_sampling), method, float(mean), se, float(lo), float(hi),
                  mult, df, table.n_checkpoints, table.n_items, checkpoint_sampling,
                  table.design.item_sampling,
                  terms, table.rollouts(), noise, claims, list(table.dropped_items),
                  shape=shape, lo_symmetric=float(mean - mult * se), hi_symmetric=float(mean + mult * se))


def _as_table(x: Any, design: Design | None) -> Table:
    if isinstance(x, Table):
        return x
    assert design is not None, "a Design is required to collapse a long table"
    return collapse(x, design)


def _checkpoint_sampling(table: Table, declared: str | None) -> str:
    """"sampled" or "fixed" for the checkpoint axis -- inferred unless explicitly declared."""
    if declared is None:
        return "sampled" if table.n_checkpoints >= 2 else "fixed"
    assert declared in ("sampled", "fixed"), f"checkpoints must be sampled|fixed, got {declared!r}"
    if declared == "sampled" and table.n_checkpoints < 2:
        raise NotEstimable("checkpoints='sampled' needs >= 2 of them; with one checkpoint the "
                           "seed-to-seed variance is not estimable -- use checkpoints='fixed' and "
                           "claim only about this checkpoint")
    return declared


# --------------------------------------------------------------------------- intervals

def interval(obs: Any, design: Design | None = None, *, checkpoints: str | None = None,
             alpha: float = 0.05, bounds: tuple[float, float] | None = None) -> Result:
    """The 95% interval for the table's mean under its Design.

    Args:
        obs: A long table (with `design`) or an already-collapsed Table.
        design: Required when `obs` is a long table.
        checkpoints: "sampled" (>= 2 from one pipeline) or "fixed"; inferred as sampled iff
            n >= 2. Sampled adds the checkpoint and interaction terms.
        alpha: Two-sided level.
        bounds: `(lo, hi)` when the value is a RATE confined to that range -- ODCV's violation
            percentage `(0, 100)`, MMLU's 0/1 correctness `(0, 1)`. The interval is then built on
            the log-odds scale, so it is asymmetric and cannot leave the range; without it the
            interval is the symmetric `mean +/- t*SE`. This is a property of the value column, not
            of the Design (ODCV runs a rate and a severity score through the SAME Design), and it
            changes only the interval's geometry -- never the estimand, the mean, or the SE. Leave
            it None for anything unbounded or not rate-like (severity, Elo, raw scores).

    Returns:
        A Result. Which spreads are combined, and the df of the multiplier:

            checkpoints sampled, items sampled   T_A + T_B - T_C   Satterthwaite df
            checkpoints sampled, items fixed     T_A               n - 1
            checkpoints fixed,   items sampled   T_B               J - 1
            checkpoints fixed,   items fixed     within-cell noise (needs R >= 2)
    """
    table = _as_table(obs, design)
    ckpt = _checkpoint_sampling(table, checkpoints)
    t = spreads(table.values, table.item_weights)
    mu = t["mu"]
    n_checkpoints, n_items = table.n_checkpoints, table.n_items
    if ckpt == "sampled" and table.design.item_sampling == "sampled":
        se2 = t["T_A"] + t["T_B"] - t["T_C"]
        fallback = se2 <= 0
        if fallback:
            se2 = max(t["T_A"], t["T_B"])
        df = satterthwaite([(t["T_A"], n_checkpoints - 1), (t["T_B"], n_items - 1),
                            (t["T_C"], (n_checkpoints - 1) * (n_items - 1))])
        terms = {"T_A": t["T_A"], "T_B": t["T_B"], "T_C": t["T_C"], "negative_fallback": fallback,
                 "df_source": "Satterthwaite over T_A, T_B, T_C"}
        return _finish(table, ckpt, "T_A + T_B - T_C, t_nu (Satterthwaite)", mu, se2, df, terms, alpha, bounds)
    if ckpt == "sampled":  # items fixed
        return _finish(table, ckpt, "T_A (per-model rates), t_{n-1}", mu, t["T_A"], table.n_checkpoints - 1,
                       {"T_A": t["T_A"]}, alpha, bounds)
    if table.design.item_sampling == "sampled":  # models fixed
        return _finish(table, ckpt, "spread of per-item rates over J (T_B), t_{J-1}", mu, t["T_B"],
                       table.n_items - 1, {"T_B": t["T_B"]}, alpha, bounds)
    # both fixed: rollouts are the only randomness
    if not np.isfinite(table.within_cell_var).all():
        raise NotEstimable(
            "checkpoints and items both fixed: the only randomness is the rollouts, and with one "
            "rollout in some cell their variance cannot be estimated. Either run >= 2 rollouts "
            "per cell, or treat items as sampled (item_sampling='sampled'), a different claim.")
    parts = _within_cell_parts(table)
    se2 = sum(v for v, _ in parts)
    return _finish(table, ckpt, "within-cell rollout noise only, t_nu (Satterthwaite over cells)",
                   mu, se2, satterthwaite(parts), {"noise_term": se2}, alpha, bounds)


def difference(obs_a: Any, obs_b: Any, design: Design | None = None, *, checkpoints: str | None = None,
               paired_checkpoints: bool = False, alpha: float = 0.05) -> Result:
    """Interval for mean(A) - mean(B), paired on every axis the two arms share.

    Deliberately has no `bounds`: a difference of two rates lives on [-span, span], can be
    negative, and has no log-odds. Nor does it need one -- the skew `bounds` corrects comes from
    the estimate approaching a boundary, and a difference sits in the middle of its range. (The
    `paired_checkpoints` branch routes a difference table through `interval` without bounds for
    the same reason.)

    Items are always paired (both arms must cover the same items; the intersection is used
    and the rest recorded). Checkpoints are paired only when `paired_checkpoints` -- the same
    checkpoints under two conditions (e.g. mandated vs incentivized), in which case the
    per-cell difference table goes through `interval` directly. Otherwise the arms have
    different checkpoints and their T_A terms add.

    Args:
        obs_a, obs_b: Long tables or Tables for the two arms.
        design: Required for long tables.
        checkpoints: As in `interval`, applied to both arms.
        paired_checkpoints: Both arms hold the SAME checkpoints and are paired on them too.
        alpha: Two-sided level.
    """
    A, B = _as_table(obs_a, design), _as_table(obs_b, design)
    shared = [u for u in A.items if u in set(B.items)]
    assert len(shared) >= 2, "arms share fewer than two items"
    A, B = A.select_items(shared), B.select_items(shared)
    dropped = sorted(set(A.dropped_items) | set(B.dropped_items)
                     | ({*obs_a.items, *obs_b.items} - set(shared) if isinstance(obs_a, Table) else set()))

    if paired_checkpoints:
        assert A.checkpoints == B.checkpoints, \
            "paired_checkpoints needs identical checkpoint labels in both arms"
        D = Table(A.values - B.values, A.checkpoints, shared, A.counts + B.counts,
                  np.minimum(A.reps, B.reps), A.within_cell_var + B.within_cell_var, A.item_weights,
                  dropped, A.design)
        r = interval(D, checkpoints=checkpoints, alpha=alpha)
        r.estimand = "difference (A - B), paired on items and checkpoints: " + r.estimand
        return r

    ckpt = (_checkpoint_sampling(A, checkpoints) if checkpoints else
            ("sampled" if min(A.n_checkpoints, B.n_checkpoints) >= 2 else "fixed"))
    if ckpt == "sampled":
        assert A.n_checkpoints >= 2 and B.n_checkpoints >= 2, \
            "checkpoints='sampled' needs >= 2 of them in each arm"
    ta, tb = spreads(A.values, A.item_weights), spreads(B.values, B.item_weights)
    d_cols = A.values.mean(axis=0) - B.values.mean(axis=0)     # per-item difference of column means
    mean = ta["mu"] - tb["mu"]
    n_items = A.n_items
    merged = Table(np.vstack([A.values, -B.values]), A.checkpoints + B.checkpoints, shared,
                   np.vstack([A.counts, B.counts]), np.vstack([A.reps, B.reps]),
                   np.vstack([A.within_cell_var, B.within_cell_var]), A.item_weights, dropped,
                   A.design)   # only for rollout/claims bookkeeping
    if ckpt == "sampled" and A.design.item_sampling == "sampled":
        t_bd = float(d_cols.var(ddof=1) / n_items)
        se2 = ta["T_A"] + tb["T_A"] + t_bd - ta["T_C"] - tb["T_C"]
        fallback = se2 <= 0
        if fallback:
            se2 = ta["T_A"] + tb["T_A"] + t_bd
        terms = {"T_A_a": ta["T_A"], "T_A_b": tb["T_A"], "T_B_d": t_bd, "T_C_a": ta["T_C"],
                 "T_C_b": tb["T_C"], "negative_fallback": fallback,
                 "df_source": "Satterthwaite over the five terms"}
        df = satterthwaite([(ta["T_A"], A.n_checkpoints - 1), (tb["T_A"], B.n_checkpoints - 1),
                            (t_bd, n_items - 1),
                            (ta["T_C"], (A.n_checkpoints - 1) * (n_items - 1)),
                            (tb["T_C"], (B.n_checkpoints - 1) * (n_items - 1))])
        r = _finish(merged, ckpt, "T_A^A + T_A^B + T_B^(d) - T_C^A - T_C^B, t_nu (Satterthwaite)",
                    mean, se2, df, terms, alpha)
    elif ckpt == "sampled":  # items fixed: Welch's two-sample t on the per-model rates
        ra = (A.values * A.item_weights[None, :]).sum(axis=1)   # see spreads on `@`
        rb = (B.values * B.item_weights[None, :]).sum(axis=1)
        va, vb = float(ra.var(ddof=1) / A.n_checkpoints), float(rb.var(ddof=1) / B.n_checkpoints)
        df = satterthwaite([(va, A.n_checkpoints - 1), (vb, B.n_checkpoints - 1)])   # Welch-Satterthwaite
        r = _finish(merged, ckpt, "Welch two-sample t on per-model rates, t_nu (Welch-Satterthwaite)",
                    mean, va + vb, df, {"var_a": va, "var_b": vb}, alpha)
    elif A.design.item_sampling == "sampled":  # models fixed, units random: paired on items
        se2 = float(d_cols.var(ddof=1) / n_items)
        r = _finish(merged, ckpt, "spread of per-item differences over J, t_{J-1}", mean, se2,
                    n_items - 1,
                    {"T_B_d": se2}, alpha)
    else:  # both fixed
        if not (np.isfinite(A.within_cell_var).all() and np.isfinite(B.within_cell_var).all()):
            raise NotEstimable("both fixed: rollout noise not estimable with one rollout per cell")
        parts = _within_cell_parts(A) + _within_cell_parts(B)
        se2 = sum(v for v, _ in parts)
        r = _finish(merged, ckpt, "within-cell rollout noise only, t_nu (Satterthwaite over cells)",
                    mean, se2, satterthwaite(parts), {"noise_term": se2}, alpha)
    def who(t: Table) -> str:
        return ("a checkpoint from its pipeline" if ckpt == "sampled"
                else ("its checkpoint" if t.n_checkpoints == 1 else f"the mean of its {t.n_checkpoints} checkpoints"))
    where = (f"a {A.design.item} drawn like these" if A.design.item_sampling == "sampled"
             else f"these {n_items} {A.design.item}s")
    r.estimand = (f"difference (A - B), paired on items: A's {who(A)} minus B's {who(B)}, "
                  f"on {where}")
    r.n_checkpoints = A.n_checkpoints + B.n_checkpoints
    return r


# --------------------------------------------------------------------------- bootstrap + binomial

def cluster_bootstrap(obs: Any, statistic: Callable[[Any], float], design: Design | None = None,
                      *, checkpoints: str | None = None, on: str = "table", n_boot: int = 10_000,
                      seed: int = 0, alpha: float = 0.05) -> dict[str, Any]:
    """Percentile bootstrap of `statistic(...)` resampling the Design's random axes.

    Units (columns) are resampled with replacement, each carrying its whole cell -- never
    rollouts or enumerated levels. When checkpoints are sampled and n >= 2, rows go too.
    Use only for statistics without a closed-form SE; for a mean, `interval` is exact, and a
    test asserts the two agree there.

    Args:
        obs: A long table, or a collapsed Table (only with `on="table"`).
        statistic: What to bootstrap. With `on="table"` it receives the resampled
            `(n_checkpoints, n_items)` matrix of cell means. With `on="rows"` it receives the
            resampled LONG rows as a DataFrame -- needed by any statistic that cannot be
            computed from cell means: a Bradley-Terry fit needs the individual battles and
            their style covariates, a median needs the raw values.
        design: Required when `obs` is a long table.
        checkpoints: "sampled"/"fixed" as in `interval`; inferred as sampled iff n >= 2.
        on: "table" (default) or "rows". See `statistic`.
        n_boot, seed, alpha: Resamples, RNG seed, two-sided level.

    Returns:
        `mean` (the statistic on the observed data), `lo`, `hi`, `se`, `method`, and the
        axes that were resampled. A unit drawn twice contributes its rows twice, which is
        what "this unit was sampled twice" means for a downstream fit.
    """
    assert on in ("table", "rows"), f"on must be table|rows, got {on!r}"
    table = _as_table(obs, design)
    ckpt = _checkpoint_sampling(table, checkpoints)
    rng = np.random.default_rng(seed)
    n_checkpoints, n_items = table.n_checkpoints, table.n_items

    if on == "rows":
        assert not isinstance(obs, Table), "on='rows' needs the long table, not a collapsed Table"
        df = _frame(obs)
        d = table.design
        # Row positions per (model, unit), so a resample is one concatenate + one .iloc
        # rather than a DataFrame slice per cell.
        pos = {k: np.asarray(v) for k, v in df.groupby([d.checkpoint, d.item]).indices.items()}
        keep = {(m, u) for m in table.checkpoints for u in table.items if (m, u) in pos}
        take_all = np.concatenate([pos[k] for k in sorted(keep, key=str)]) if keep else np.array([], int)
        point = float(statistic(df.iloc[np.sort(take_all)]))
    else:
        point = float(statistic(table.values))

    draws = np.empty(n_boot)
    for b in range(n_boot):
        cols = (rng.integers(0, n_items, n_items) if table.design.item_sampling == "sampled"
                else np.arange(n_items))
        rows = (rng.integers(0, n_checkpoints, n_checkpoints) if ckpt == "sampled"
                else np.arange(n_checkpoints))
        if on == "table":
            draws[b] = statistic(table.values[np.ix_(rows, cols)])
        else:
            take = [pos[(table.checkpoints[i], table.items[j])] for i in rows for j in cols
                    if (table.checkpoints[i], table.items[j]) in pos]
            draws[b] = statistic(df.iloc[np.concatenate(take)] if take else df.iloc[:0])
    return {"mean": point, "lo": float(np.quantile(draws, alpha / 2)),
            "hi": float(np.quantile(draws, 1 - alpha / 2)), "se": float(draws.std(ddof=1)),
            "method": f"cluster bootstrap over {'units' if ckpt == 'fixed' else 'models and units'}, "
                      f"{n_boot} resamples, statistic on the {'cell table' if on == 'table' else 'long rows'}",
            "n_boot": n_boot, "models": ckpt, "units": table.design.item_sampling, "on": on}


def wilson(k: float, n: int, z: float = Z_95) -> tuple[float, float]:
    """Wilson score interval for a proportion k/n.

    Wilson rather than the naive `p +/- z*sqrt(p(1-p)/n)`: it stays inside [0, 1] and keeps
    a sensible width at the edges, where the naive interval has width zero at 0/n and can
    run below zero nearby. Use it for a rate near 0 or 1, where the symmetric intervals
    above stop making sense. Bounds are clamped: rounding can otherwise put them a
    floating-point hair outside [0, 1], which reads as 100.000000001% in a report.
    """
    assert n > 0
    p = k / n
    denom = 1 + z ** 2 / n
    centre = (p + z ** 2 / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value from the two discordant counts."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2.0 ** n
    return min(1.0, 2.0 * tail)


# --------------------------------------------------------------------------- t quantile (no scipy)

def _betacf(a: float, b: float, x: float) -> float:
    """Continued fraction for the incomplete beta function (Lentz)."""
    tiny = 1e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    d = 1.0 / (d if abs(d) > tiny else tiny)
    h = d
    for m in range(1, 300):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > tiny else tiny)
        c = 1.0 + aa / (c if abs(c) > tiny else tiny)
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > tiny else tiny)
        c = 1.0 + aa / (c if abs(c) > tiny else tiny)
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-14:
            break
    return h


def _betainc(a: float, b: float, x: float) -> float:
    """Regularised incomplete beta I_x(a, b)."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    ln_beta = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
    front = math.exp(ln_beta + a * math.log(x) + b * math.log(1.0 - x))
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - front * _betacf(b, a, 1.0 - x) / b


def t_cdf(t: float, df: float) -> float:
    if math.isinf(df):
        return NormalDist().cdf(t)
    x = df / (df + t * t)
    p = 0.5 * _betainc(df / 2.0, 0.5, x)
    return 1.0 - p if t > 0 else p


def t_quantile(p: float, df: float) -> float:
    """Quantile of Student's t (df may be inf -> normal). Bisection on the CDF; no scipy."""
    assert 0 < p < 1
    if math.isinf(df):
        return NormalDist().inv_cdf(p)
    assert df > 0
    if p < 0.5:
        return -t_quantile(1 - p, df)
    lo, hi = 0.0, 1.0
    while t_cdf(hi, df) < p:
        hi *= 2.0
        if hi > 1e6:
            break
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if t_cdf(mid, df) < p:
            lo = mid
        else:
            hi = mid
        if hi - lo < 1e-10:
            break
    return 0.5 * (lo + hi)
