# ABOUTME: The per-task trace band every trajectory set is generated and split against
# ABOUTME: Import from here: from src.trajectory_data.dataset_shape import MIN_TRACES_PER_TASK
"""
Dataset shape policy, in one place.

C_theta is trained by contrast between an expert and an unsafe set and split by
task id, so the *shape* of those sets is what it actually learns:

  * below MIN_TRACES_PER_TASK, a task contributes too few examples for the
    held-out split to say anything about it;
  * above MAX_TRACES_PER_TASK, near-duplicate traces of one task dominate the
    loss and C_theta learns to recognise that task rather than the safety
    boundary. Not hypothetical — the 2026-08-17 expert set was 110 traces of
    task 237.

The band applies per task AND per set: expert and unsafe are each generated to
the same target on the same task ids, so class is never confounded with task
coverage. Widen a dataset by adding task ids, never by raising the ceiling.

Enforced at both ends of the pipeline:
  * generation — src/trajectory_generation/generation_runner.py rejects an
    out-of-band target and stops generating a task once it reaches it;
  * splitting  — scripts/make_demo_splits.py downsamples any task above the
    ceiling and warns about any task below the floor.
"""
from __future__ import annotations

#: Fewest verified traces a task may contribute to a set.
MIN_TRACES_PER_TASK = 5

#: Most verified traces a task may contribute to a set.
MAX_TRACES_PER_TASK = 10


def describe_band() -> str:
    """Human-readable band, for error messages and CLI help."""
    return f"{MIN_TRACES_PER_TASK}-{MAX_TRACES_PER_TASK} traces per task"


def check_target(target: int, source: str) -> int:
    """
    Validate a configured per-task target, or raise.

    `source` names the setting in the message so a bad config points at itself.
    """
    if not MIN_TRACES_PER_TASK <= target <= MAX_TRACES_PER_TASK:
        raise ValueError(
            f"{source}={target} is outside the band ({describe_band()}). "
            "Add task ids to widen coverage; more traces of the same task will not.")
    return target
