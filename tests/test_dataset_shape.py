# ABOUTME: Tests the per-task trace band — the generation target and the split-time cap that enforce it
# ABOUTME: Run: pytest tests/test_dataset_shape.py -q
"""
The band exists because the shape of the contrast set is what C_theta learns.

Generation aims at a per-task target and splitting caps anything that still
arrives over it, so no single task can supply most of the training signal. The
2026-08-17 expert set — 110 traces of one task — is the failure these guard.
"""
from __future__ import annotations

import importlib.util
import random
from pathlib import Path

import pytest

from src.trajectory_data.dataset_shape import (MAX_TRACES_PER_TASK,
                                               MIN_TRACES_PER_TASK,
                                               check_target, describe_band)

REPO_ROOT = Path(__file__).resolve().parent.parent


def _load_splits_module():
    """Import scripts/make_demo_splits.py, which is a CLI rather than a package."""
    spec = importlib.util.spec_from_file_location(
        "make_demo_splits", REPO_ROOT / "scripts" / "make_demo_splits.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ── The band itself ───────────────────────────────────────────────────────────

def test_band_is_a_real_range():
    assert 1 < MIN_TRACES_PER_TASK < MAX_TRACES_PER_TASK
    assert describe_band() == f"{MIN_TRACES_PER_TASK}-{MAX_TRACES_PER_TASK} traces per task"


def test_check_target_accepts_the_band_and_rejects_the_rest():
    for good in range(MIN_TRACES_PER_TASK, MAX_TRACES_PER_TASK + 1):
        assert check_target(good, "test") == good
    for bad in (0, 1, MIN_TRACES_PER_TASK - 1, MAX_TRACES_PER_TASK + 1, 110):
        with pytest.raises(ValueError, match="outside the band"):
            check_target(bad, "generation_loop.traces_per_task")


def test_rejection_names_the_offending_setting():
    with pytest.raises(ValueError, match="generation_loop.traces_per_task=110"):
        check_target(110, "generation_loop.traces_per_task")


# ── The split-time cap ────────────────────────────────────────────────────────

def _rows(counts: dict[str, int]) -> list[dict]:
    return [{"task_instance_id": task, "n": i}
            for task, n in counts.items() for i in range(n)]


def test_over_represented_task_is_downsampled_to_the_cap():
    splits = _load_splits_module()
    rows = _rows({"237": 110, "244": 6})

    kept, dropped = splits.cap_per_task(rows, MAX_TRACES_PER_TASK, random.Random(0))

    by_task: dict[str, int] = {}
    for r in kept:
        by_task[r["task_instance_id"]] = by_task.get(r["task_instance_id"], 0) + 1
    assert by_task == {"237": MAX_TRACES_PER_TASK, "244": 6}
    assert dropped == {"237": 110 - MAX_TRACES_PER_TASK}


def test_cap_samples_rather_than_truncates():
    """Traces are written in generation order, so the first N share a pass, a
    seed and often a plan — taking a prefix would keep the least diverse ones."""
    splits = _load_splits_module()
    rows = _rows({"237": 50})

    kept, _ = splits.cap_per_task(rows, MAX_TRACES_PER_TASK, random.Random(0))

    assert [r["n"] for r in kept] != list(range(MAX_TRACES_PER_TASK))


def test_a_balanced_set_passes_through_untouched():
    splits = _load_splits_module()
    rows = _rows({"237": 8, "244": 8, "246": 5})

    kept, dropped = splits.cap_per_task(rows, MAX_TRACES_PER_TASK, random.Random(0))

    assert len(kept) == len(rows)
    assert dropped == {}
