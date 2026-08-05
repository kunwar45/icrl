"""
Load demos from either format the project produces.

Two writers, two shapes:

  scripts/demos/collect_safe_trajectories.py  → one `task_<id>_trace_<n>.json` per
      trajectory in a directory (this is what the SLURM jobs write to
      $SCRATCH/trajectories/safe)
  the older collectors                  → a single `*.jsonl` of Trajectory dicts
      (data/demos/safe.jsonl, unsafe.jsonl, webarena_raw.jsonl)

Everything downstream wants `Trajectory`, so the conversion lives here rather
than being re-implemented per script.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import List

from src.data.actions import normalize_safety_report
from src.data.trajectory import Step, Trajectory

logger = logging.getLogger(__name__)

TRACE_GLOB = "task_*_trace_*.json"

# Actions that mean the agent decided it was finished, as opposed to running out
# of steps. Used only when a trace predates the `terminated` field.
_TERMINAL_ACTIONS = ("answer(", "report_infeasible(")


def _infer_terminated(payload: dict) -> bool:
    """
    Recover the `terminated` flag for traces written before it was recorded.

    Two signals, both conservative: the benchmark ends the episode on success,
    so reward 1.0 implies termination; otherwise the agent must have called a
    terminal action itself. Anything else is treated as truncated, which is the
    safer error — it costs a demo rather than admitting a padded one.
    """
    if payload.get("terminated") is not None:
        return bool(payload["terminated"])
    if float(payload.get("reward") or 0.0) >= 1.0:
        return True
    steps = payload.get("steps") or []
    if steps:
        last = str(steps[-1].get("action", ""))
        return any(last.startswith(a) or f" {a}" in last for a in _TERMINAL_ACTIONS)
    return False


def trace_json_to_trajectory(payload: dict, source_file: str = "") -> Trajectory:
    """Convert one `task_<id>_trace_<n>.json` payload to a Trajectory."""
    reports = [normalize_safety_report(r) for r in payload.get("safety_report", [])]
    violated = [r for r in reports if r["violated"]]

    steps = [
        Step(
            step_idx=int(s.get("step_idx", i)),
            action=str(s.get("action", "")),
            observation=str(s.get("observation", "")),
        )
        for i, s in enumerate(payload.get("steps", []))
    ]

    task_id = str(payload.get("task_id", "?"))
    stem = Path(source_file).stem if source_file else f"task_{task_id}"

    return Trajectory(
        trajectory_id=stem,
        task_type=str(payload.get("task_type") or "suitecrm"),
        task_instance_id=task_id,
        steps=steps,
        is_safe=len(violated) == 0,
        source=str(payload.get("model", "unknown")),
        reward=float(payload.get("reward") or 0.0),
        terminated=_infer_terminated(payload),
    )


def load_trace_dir(directory: Path) -> List[Trajectory]:
    """Load every `task_*_trace_*.json` in a directory (non-recursive)."""
    files = sorted(directory.glob(TRACE_GLOB))
    trajectories = []
    for path in files:
        try:
            payload = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as e:
            logger.warning("Skipping %s: %s", path, e)
            continue
        trajectories.append(trace_json_to_trajectory(payload, source_file=str(path)))
    logger.info("Loaded %d trace files from %s", len(trajectories), directory)
    return trajectories


def load_jsonl(path: Path) -> List[Trajectory]:
    trajectories = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                trajectories.append(Trajectory.from_dict(json.loads(line)))
    return trajectories


def load_demos(path: str | Path) -> List[Trajectory]:
    """
    Load demos from a `.jsonl` file or a directory of trace JSONs.

    Accepting both is what lets `--safe-demos $SCRATCH/trajectories/safe` work
    directly against what the SLURM collection job actually wrote.
    """
    path = Path(path)
    if path.is_dir():
        trajectories = load_trace_dir(path)
        if not trajectories:
            raise FileNotFoundError(
                f"No {TRACE_GLOB} files in {path}. Point --safe-demos at the "
                f"collection output directory or a .jsonl file."
            )
        return trajectories
    if not path.exists():
        raise FileNotFoundError(f"{path} not found.")
    return load_jsonl(path)
