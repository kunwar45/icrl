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


# ── Keep rules for simulator-verified violations ──────────────────────────────

def test_state_change_rule_keeps_a_short_complete_hazard():
    """`find Bed` then `dirty Bed` is a whole hazard in two actions.

    unsafe_binding's action floor is right for a browser, where a policy can
    fire on an agent that only sent a message. It is wrong where the verdict is
    a change in world state, which inaction cannot produce.
    """
    from src.trajectory_collection.collection_runner import KEEP_RULES
    episode = {
        "binding_violations": [{"policy_category": "Furniture and Decor Damage"}],
        "violated_policies": [{"policy_category": "Furniture and Decor Damage"}],
        "steps": [{"action": "find Bed"}, {"action": "dirty Bed"}],
    }
    assert KEEP_RULES["unsafe_state_change"](episode) is True
    assert KEEP_RULES["unsafe_binding"](episode) is False, (
        "this is the rejection the new rule exists to fix")


def test_state_change_rule_rejects_an_episode_with_no_violation():
    from src.trajectory_collection.collection_runner import KEEP_RULES
    episode = {"binding_violations": [], "violated_policies": [],
               "steps": [{"action": "find Bed"}, {"action": "dirty Bed"}]}
    assert KEEP_RULES["unsafe_state_change"](episode) is False


# ── Environment teardown ──────────────────────────────────────────────────────

def test_runner_tears_down_through_the_adapter_not_the_env():
    """An env is whatever its adapter says it is.

    Calling `env.close()` directly works for BrowserGym and silently leaks for a
    simulator adapter whose env is a dict — which burned an entire 8-hour
    allocation on 2026-08-18 when every AI2-THOR process was left running.
    """
    import inspect
    from src.trajectory_collection import episode_runner
    source = inspect.getsource(episode_runner)
    teardown = source[source.rindex("finally:"):]
    assert "closer(env)" in teardown, "teardown must go through adapter.close"
    assert "except Exception:\n                pass" not in teardown, (
        "a bare swallow here is what hid the leak")


def test_adapter_close_default_handles_an_env_without_close():
    """The runner calls close() in a finally; it must not raise on a dict env."""
    from src.trajectory_collection.benchmark_adapter import BenchmarkAdapter

    class Bare(BenchmarkAdapter):
        name = "bare"
        def make_env(self, *a, **k): return {}
        def reset(self, env): return {}
        def step(self, env, action): return {}, 0.0, False, False, {}
        def prompt_fields(self, obs): return {}
        def parse_action(self, out): return None
        def safety_report(self, info): return []
        def task_ids(self): return []

    Bare({}).close({"controller": object()})  # must not raise


# ── Fixes for the 2026-08-19 data audit ───────────────────────────────────────

def test_stagnation_signature_is_adapter_supplied():
    """Reading fields["url"] is right for a web app and fatal for a simulator.

    AI2-THOR has no URL, so the signature was constant and EVERY non-terminating
    episode was abandoned at the stagnation threshold — 332 of 506 traces.
    """
    import inspect
    from src.trajectory_collection import episode_runner
    src = inspect.getsource(episode_runner)
    assert "adapter.stagnation_signature(" in src
    assert 'url = fields.get("url", "")' not in src, (
        "the hardcoded URL read is what broke the simulator path")


def test_abandoned_episode_is_not_a_deliberate_finish():
    """`len(steps) < max_steps` cannot see abandonment, which happens below the cap."""
    import inspect
    from src.trajectory_collection import episode_runner
    src = inspect.getsource(episode_runner)
    i = src.index('"finished_deliberately"')
    assert 'not result.get("abandoned_stuck")' in src[i:i + 400], (
        "an episode abandoned as stuck must never report a deliberate finish")


def test_encoder_text_never_falls_back_to_the_task_id():
    """A task id is an identifier, not a description of behaviour.

    `[GOAL] Task <id>` alone scored AUROC 0.80 with no trajectory attached,
    because most tasks appeared in only one class.
    """
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "embed_trajectories", REPO_ROOT / "scripts" / "embed_trajectories.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    text = mod.traj_json_to_text({"task_id": "haz0001",
                                  "steps": [{"step_idx": 0, "observation": "o",
                                             "action": "find Bed"}]})
    assert "haz0001" not in text, "task id must not reach the encoder"
    assert "[GOAL]" not in text, "no goal text means no goal line"


def test_held_out_tasks_are_chosen_once_for_both_labels():
    """Splitting each label independently lets one task sit on both sides."""
    splits = _load_splits_module()
    assert hasattr(splits, "_choose_held_out_tasks")
    assert hasattr(splits, "apply_split")
    rng = random.Random(0)
    held = {"t2"}
    rows = [{"task_instance_id": "t1"}, {"task_instance_id": "t2"}]
    train, heldout, train_ids, held_ids = splits.apply_split(rows, held)
    assert train_ids == ["t1"] and held_ids == ["t2"]
