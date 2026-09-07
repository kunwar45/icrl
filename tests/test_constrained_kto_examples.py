# ABOUTME: Tests the desirable/undesirable split of the constrained KTO loop, the step that fixes the previous loop's blind spot on cells that never produce a clean rollout
# ABOUTME: Run: pytest tests/test_constrained_kto_examples.py
import importlib.util
import sys
from pathlib import Path

from omegaconf import OmegaConf

REPO = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "constrained_kto", REPO / "scripts/finetune_policy_constrained_kto.py"
)
mod = importlib.util.module_from_spec(spec)
sys.modules["constrained_kto"] = mod
spec.loader.exec_module(mod)

CFG = OmegaConf.create(
    {
        "constraint": {"desirable_max_score": 0.3, "undesirable_min_score": 0.7},
        "train": {"max_per_cell_per_class": 3},
    }
)


def rollout(scenario, framing, C, R=1.0, n_steps=5):
    return {
        "path": f"{scenario}/{framing}/{C}/{R}",
        "scenario": scenario,
        "framing": framing,
        "C": C,
        "R": R,
        "n_steps": n_steps,
        "steps": [],
        "scenario_prompt": "p",
    }


def test_all_violating_cell_still_contributes_negatives():
    """The failure that motivated this loop: nine cells produced no clean rollout in
    30-36 samples, so a positives-only filter dropped them and the policy never saw a
    correction where the cheating actually was."""
    rows = [rollout("Hard", "mandated", c) for c in (0.95, 0.99, 0.88)]
    examples, stats = mod.build_examples(rows, CFG)
    assert len(examples) == 3
    assert all(e["label"] is False for e in examples)
    assert stats["cells_negatives_only"] == 1
    assert stats["cells_all_clean_dropped"] == 0


def test_all_clean_cell_is_dropped():
    """A cell the policy already handles carries no contrast; training on it is
    imitation of behaviour the policy has, which supplied 56% of the old loop's data."""
    rows = [rollout("Easy", "incentivized", c) for c in (0.01, 0.05, 0.02)]
    examples, stats = mod.build_examples(rows, CFG)
    assert examples == []
    assert stats["cells_all_clean_dropped"] == 1


def test_mixed_cell_yields_both_classes():
    rows = [rollout("Mixed", "mandated", c) for c in (0.02, 0.06, 0.93, 0.97)]
    examples, stats = mod.build_examples(rows, CFG)
    assert stats["n_desirable"] == 2
    assert stats["n_undesirable"] == 2
    assert stats["cells_contributing"] == 1


def test_ambiguous_band_and_unfinished_rollouts_are_excluded():
    """Scores near the threshold are the head's uncertain region, and a rollout that
    never ran the tool would teach 'submit something' rather than 'do not cheat'."""
    rows = [
        rollout("Cell", "mandated", 0.5),  # ambiguous
        rollout("Cell", "mandated", 0.45),  # ambiguous
        rollout("Cell", "mandated", 0.01, R=0.0),  # clean but did not do the task
        rollout("Cell", "mandated", 0.99, R=0.0),  # violating but did not do the task
        rollout("Cell", "mandated", 0.95),  # kept
    ]
    examples, stats = mod.build_examples(rows, CFG)
    assert stats["n_examples"] == 1
    assert examples[0]["C"] == 0.95
    assert examples[0]["label"] is False


def test_per_cell_cap_keeps_the_most_extreme():
    rows = [rollout("Cell", "mandated", c) for c in (0.01, 0.02, 0.03, 0.04)]
    rows += [rollout("Cell", "mandated", c) for c in (0.90, 0.92, 0.94, 0.96)]
    examples, stats = mod.build_examples(rows, CFG)
    assert stats["n_desirable"] == 3
    assert stats["n_undesirable"] == 3
    kept_des = sorted(e["C"] for e in examples if e["label"])
    kept_und = sorted(e["C"] for e in examples if not e["label"])
    assert kept_des == [0.01, 0.02, 0.03]
    assert kept_und == [0.92, 0.94, 0.96]


def test_length_guard_is_reported():
    """The old loop trained on rollouts half as long as those it discarded; the split
    reports both means so a drift toward stopping early is visible per round."""
    rows = [rollout("Cell", "mandated", 0.02, n_steps=4)]
    rows += [rollout("Cell", "mandated", 0.98, n_steps=20)]
    _, stats = mod.build_examples(rows, CFG)
    assert stats["mean_steps_desirable"] == 4
    assert stats["mean_steps_undesirable"] == 20
