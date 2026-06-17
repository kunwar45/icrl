"""
AgentLab/BrowserGym trace loader.

Converts AgentLab ExpResult objects (structured BrowserGym traces) into the
project's Trajectory format for use in ICRL training.

Replaces the old webarena_loader.py Playwright trace parser.  The key
improvement: AgentLab already stores (AXTree text, action) pairs per step —
no manual DOM extraction or selector mapping needed.
"""
from __future__ import annotations

import logging
import uuid
from pathlib import Path
from typing import Optional

from src.data.trajectory import Step, Trajectory

logger = logging.getLogger(__name__)


def agentlab_obs_to_text(obs: dict) -> str:
    """Convert a BrowserGym observation dict to the GOAL/URL/PAGE text format.

    Produces the same "GOAL:\\nURL:\\nPAGE:\\n" format as st_webagent.obs_repr()
    so TrajectoryEncoder receives identical input regardless of collection path.
    """
    # Goal from chat_messages (first user message)
    goal = ""
    for msg in obs.get("chat_messages", []):
        if isinstance(msg, dict) and msg.get("role") == "user":
            goal = msg.get("message", msg.get("content", ""))
            break

    # URL: try direct key, then open_pages_urls list
    url = obs.get("url", "")
    if not url:
        pages = obs.get("open_pages_urls", [])
        if pages:
            active = obs.get("active_page_index", {})
            idx = active.get("value", 0) if isinstance(active, dict) else 0
            url = pages[idx] if idx < len(pages) else pages[0]

    # AXTree text: prefer pre-flattened string, fall back to flattening the object
    axtree_text = obs.get("axtree_txt", "")
    if not axtree_text:
        axtree_obj = obs.get("axtree_object", "")
        if axtree_obj:
            try:
                from browsergym.utils.obs import flatten_axtree_to_str
                axtree_text = flatten_axtree_to_str(axtree_obj)
            except Exception:
                axtree_text = str(axtree_obj)[:4000]

    return f"GOAL:\n{goal}\nURL:\n{url}\nPAGE:\n{axtree_text}"


def exp_result_to_trajectory(
    exp_result,
    task_id: Optional[str] = None,
) -> Optional[Trajectory]:
    """Convert one BrowserGym ExpResult to a Trajectory.

    Returns None if the result has no steps (e.g. env crash before first action).
    is_safe is left False — caller must run SafetyVerifier or ground-truth eval.
    """
    steps_info = getattr(exp_result, "steps_info", None) or []
    steps: list[Step] = []

    for step_idx, step_info in enumerate(steps_info):
        action = getattr(step_info, "action", "") or ""
        obs_dict = (
            getattr(step_info, "obs", None)
            or getattr(step_info, "observation", None)
            or {}
        )
        obs_text = agentlab_obs_to_text(obs_dict) if obs_dict else ""
        steps.append(Step(step_idx=step_idx, action=action, observation=obs_text))

    if not steps:
        return None

    # Termination flag and cumulative reward
    terminated = bool(
        getattr(exp_result, "terminated", False)
        or (
            isinstance(getattr(exp_result, "summary_info", None), dict)
            and exp_result.summary_info.get("terminated", False)
        )
    )
    reward = float(
        getattr(exp_result, "cum_reward", None)
        or (
            isinstance(getattr(exp_result, "summary_info", None), dict)
            and exp_result.summary_info.get("cum_reward", 0.0)
        )
        or 0.0
    )

    # Task ID: prefer caller-supplied, then env_args, then a fresh UUID
    if task_id is None:
        env_args = getattr(exp_result, "env_args", None)
        if env_args is not None:
            task_id = str(getattr(env_args, "task_name", uuid.uuid4().hex[:8]))
        else:
            task_id = uuid.uuid4().hex[:8]

    return Trajectory(
        trajectory_id=uuid.uuid4().hex[:8],
        task_type="webarena",
        task_instance_id=task_id,
        steps=steps,
        is_safe=False,
        source="agentlab",
        reward=reward,
        terminated=terminated,
    )


def load_agentlab_study(study_dir: str | Path) -> list[Trajectory]:
    """Load all experiment results from an AgentLab study directory.

    Walks the directory recursively for individual experiment subdirs, converts
    each to a Trajectory, and logs termination statistics (the key quality signal
    for ICRL demo filtering).
    """
    study_dir = Path(study_dir)
    try:
        from agentlab.analyze import inspect_results
    except ImportError as exc:
        raise ImportError(
            "agentlab is not installed. Run: pip install agentlab"
        ) from exc

    trajs: list[Trajectory] = []
    total = 0
    terminated_count = 0

    for exp_result in inspect_results.yield_all_exp_results(str(study_dir)):
        total += 1
        traj = exp_result_to_trajectory(exp_result)
        if traj is not None:
            trajs.append(traj)
            if traj.terminated:
                terminated_count += 1

    logger.info(
        "Loaded %d/%d trajectories from %s  (%d terminated, %.0f%%)",
        len(trajs),
        total,
        study_dir,
        terminated_count,
        100 * terminated_count / max(total, 1),
    )
    return trajs


def download_hf_webarena_traces(
    output_dir: str | Path,
    dataset_id: str,
    hf_token: Optional[str] = None,
) -> Path:
    """Download pre-recorded BrowserGym traces from HuggingFace Hub.

    Returns the local path of the downloaded snapshot, ready for
    load_agentlab_study().
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise ImportError(
            "huggingface-hub is not installed. Run: pip install huggingface-hub"
        ) from exc

    logger.info("Downloading BrowserGym traces: %s → %s", dataset_id, output_dir)
    path = snapshot_download(
        repo_id=dataset_id,
        repo_type="dataset",
        local_dir=str(output_dir),
        token=hf_token,
    )
    logger.info("Downloaded to: %s", path)
    return Path(path)
