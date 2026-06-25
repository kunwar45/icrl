"""
ST-WebAgentBench interface and BrowserGym environment factory.

ST-WebAgentBench repo: https://github.com/segev-shlomov/ST-WebAgentBench
Install: pip install st-webagentbench  (or clone + pip install -e .)
"""
from __future__ import annotations

import glob
import json
import logging
import os
from typing import Dict, List, Optional

from src.data.trajectory import Trajectory, Step

logger = logging.getLogger(__name__)

TASK_TYPES = [
    "suitecrm",
    "gitlab",
    "shopping",
    "shopping_admin",
    "reddit",
    "wikipedia",
    # legacy type names kept for backward compat
    "delete_record",
    "expose_private_field",
    "irreversible_submit",
    "skip_confirmation",
    "bulk_destructive_action",
]

_ENV_ID_TEMPLATE = "browsergym/STWebAgentBenchEnv.{task_id}"

# Keys to keep from raw BrowserGym observations (strips screenshot bytes)
_OBS_KEYS_TO_KEEP = {"goal", "url", "axtree_object", "policies", "chat_messages", "pruned_html"}


# ── Observation helpers (copied from gridworld/icrl/envs/stwebagent.py) ───────

def _strip_screenshot(obs: dict) -> dict:
    return {k: v for k, v in obs.items() if k in _OBS_KEYS_TO_KEEP}


def _format_policies(policies) -> str:
    if not policies:
        return "None"
    if isinstance(policies, list):
        return "\n".join(f"- {p}" for p in policies)
    return str(policies)


def _flatten_axtree(axtree, max_chars: int = 4000) -> str:
    if axtree is None:
        return ""
    try:
        from browsergym.utils.obs import flatten_axtree_to_str
        text = flatten_axtree_to_str(axtree)
    except Exception:
        text = str(axtree)
    return text[:max_chars] + "\n…[truncated]" if len(text) > max_chars else text


def obs_repr(obs: dict) -> str:
    """Format a BrowserGym observation as a text string for the LLM."""
    url = obs.get('url', '') or (obs.get('open_pages_urls') or [''])[0]
    goal = obs.get('goal', '')
    if not goal:
        for msg in obs.get('chat_messages', []):
            if msg.get('role') == 'user':
                goal = msg.get('message', '')
                break
    return (
        f"GOAL: {goal}\n"
        f"URL: {url}\n"
        f"POLICIES:\n{_format_policies(obs.get('policies', []))}\n"
        f"PAGE (use [bid] numbers for actions):\n{_flatten_axtree(obs.get('axtree_object'))}"
    )


# ── Task dict normalization ───────────────────────────────────────────────────

def _normalize_task_dict(raw: dict, filename_stem: str = "") -> Optional[dict]:
    """Map a raw benchmark JSON object to the canonical schema."""
    task_id = str(raw.get("task_id") or raw.get("id") or filename_stem)
    if not task_id:
        return None

    # Derive task_type: prefer explicit field, fall back to first site name
    task_type = str(raw.get("task_type") or raw.get("type") or raw.get("category") or "")
    if not task_type:
        sites = raw.get("sites", [])
        task_type = sites[0] if sites else "unknown"

    # Filter: skip tasks whose type isn't one we recognise (if TASK_TYPES is populated)
    if TASK_TYPES and task_type not in TASK_TYPES:
        return None

    return {
        "task_id": task_id,
        "task_type": task_type,
        "constraint_description": str(
            raw.get("constraint_description") or raw.get("intent") or raw.get("description") or f"Task {task_id}"
        ),
        "ground_truth_label": bool(raw.get("ground_truth_label", True)),
        "policies": list(raw.get("policies") or []),
        "goal": str(raw.get("intent") or raw.get("goal") or ""),
        "start_url": str(raw.get("start_url") or ""),
        "storage_state": str(raw.get("storage_state") or ""),
    }


# ── Main class ────────────────────────────────────────────────────────────────

class STWebAgentBench:
    def __init__(self, benchmark_root: str):
        self.root = benchmark_root
        self._tasks: Optional[Dict] = None

    # ── Task loading ──────────────────────────────────────────────────────────

    def load_tasks(self) -> Dict:
        """
        Load task definitions using a three-tier fallback:
          Tier 1: tasks already injected via inject_task_id_map() — instant return
          Tier 2: scan benchmark_root/tasks/*.json
          Tier 3: enumerate the BrowserGym gymnasium registry
        """
        # Tier 1: already populated (e.g. by inject_task_id_map or a prior call)
        if self._tasks is not None:
            return self._tasks

        # Tier 2a: benchmark package test.raw.json (the real task file)
        tasks = self._load_from_package_json()
        if tasks:
            logger.info(f"Loaded {len(tasks)} tasks from benchmark package")
            self._tasks = tasks
            return tasks

        # Tier 2b: JSON directory under benchmark_root
        task_dir = os.path.join(self.root, "tasks")
        if os.path.isdir(task_dir):
            tasks = self._load_from_json_dir(task_dir)
            if tasks:
                logger.info(f"Loaded {len(tasks)} tasks from {task_dir}")
                self._tasks = tasks
                return tasks

        # Tier 3: BrowserGym registry (task_type will be 'unknown')
        tasks = self._load_from_registry()
        logger.info(f"Loaded {len(tasks)} task stubs from BrowserGym registry")
        self._tasks = tasks
        return tasks

    def _load_from_package_json(self) -> Dict:
        """Load tasks from stwebagentbench.test.raw.json (installed benchmark package)."""
        try:
            import importlib.resources
            import stwebagentbench
            raw_text = importlib.resources.files(stwebagentbench).joinpath("test.raw.json").read_text()
            all_configs = json.loads(raw_text)
        except Exception as e:
            logger.debug(f"Could not load from benchmark package: {e}")
            return {}

        tasks = {}
        for raw in all_configs:
            task = _normalize_task_dict(raw)
            if task:
                tasks[task["task_id"]] = task
        return tasks

    def _load_from_json_dir(self, task_dir: str) -> Dict:
        tasks = {}
        for fpath in sorted(glob.glob(os.path.join(task_dir, "*.json"))):
            stem = os.path.splitext(os.path.basename(fpath))[0]
            try:
                with open(fpath) as f:
                    raw = json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                logger.warning(f"Skipping {fpath}: {e}")
                continue
            task = _normalize_task_dict(raw, filename_stem=stem)
            if task:
                tasks[task["task_id"]] = task
        return tasks

    def _load_from_registry(self) -> Dict:
        try:
            import gymnasium
            import browsergym  # noqa: registers envs
        except ImportError:
            logger.warning("BrowserGym not installed — cannot discover tasks from registry.")
            return {}

        tasks = {}
        prefix = "browsergym/STWebAgentBenchEnv."
        for env_id in gymnasium.envs.registry.keys():
            if env_id.startswith(prefix):
                task_id = env_id[len(prefix):]
                # Cannot determine task_type from registry alone — type is "unknown"
                tasks[task_id] = {
                    "task_id": task_id,
                    "task_type": "unknown",
                    "constraint_description": f"Task {task_id}",
                    "ground_truth_label": True,
                    "policies": [],
                }
        return tasks

    def inject_task_id_map(self, mapping: Dict[str, List[str]]) -> None:
        """
        Directly populate _tasks from a {task_type: [task_id, ...]} dict.
        Bypasses all file discovery. Called from collect_demos.py when
        cfg.collection.task_ids is non-empty.
        """
        tasks: Dict = {}
        for task_type, ids in mapping.items():
            for task_id in (ids or []):
                tid = str(task_id)
                tasks[tid] = {
                    "task_id": tid,
                    "task_type": task_type,
                    "constraint_description": f"Task of type: {task_type}",
                    "ground_truth_label": True,
                    "policies": [],
                }
        if tasks:
            self._tasks = tasks
            logger.info(f"Injected {len(tasks)} task IDs from config ({len(mapping)} types)")

    def get_tasks_by_type(self, task_type: str) -> List[Dict]:
        if self._tasks is None:
            self._tasks = self.load_tasks()
        return [t for t in self._tasks.values() if t["task_type"] == task_type]

    # ── Environment factory ───────────────────────────────────────────────────

    def env_for_task(self, task_id: str, headless: bool = True):
        """
        Return a raw BrowserGym gymnasium environment for task_id.
        Caller is responsible for reset(), step(), and close().
        """
        try:
            import gymnasium
            import browsergym  # noqa: registers envs
        except ImportError as e:
            raise ImportError(
                "BrowserGym is required for demo collection. "
                "Install: pip install -e BrowserGym/browsergym-core "
                "         pip install -e BrowserGym/stwebagentbench"
            ) from e

        env_id = _ENV_ID_TEMPLATE.format(task_id=task_id)
        # --no-sandbox and --disable-dev-shm-usage are required on SLURM compute
        # nodes where user namespaces are disabled and /dev/shm is undersized.
        # Without them Chromium crashes silently and every page renders blank.
        return gymnasium.make(
            env_id,
            headless=headless,
            pw_extra_args=[
                "--no-sandbox",
                "--disable-dev-shm-usage",
                "--disable-gpu",
            ],
        )

    # ── Benchmark evaluator ───────────────────────────────────────────────────

    def evaluate_trajectory(self, trajectory: Trajectory) -> Dict:
        """
        Run the benchmark's ground-truth evaluator on a trajectory.
        Falls back to a reward-based heuristic if the benchmark evaluator
        is not available (import fails or raises).

        # TODO (Kunwar): confirm the correct import path and function signature
        # once the benchmark is installed. Check the benchmark's __init__.py.
        """
        try:
            from browsergym.stwebagentbench import evaluate  # type: ignore
            result = evaluate(
                task_id=trajectory.task_instance_id,
                actions=[step.action for step in trajectory.steps],
            )
            return {
                "task_completion": bool(result.get("success", False)),
                "safety_violation": bool(result.get("constraint_violated", False)),
                "ground_truth_constraint_violated": bool(result.get("constraint_violated", False)),
            }
        except (ImportError, Exception) as e:
            logger.debug(f"Benchmark evaluator unavailable ({e}), using heuristic")
            completed = trajectory.reward is not None and trajectory.reward > 0
            # is_safe is set from env.info["safety_report"] during the episode — trust it
            # over the old fallback that always returned False for safety_violation
            violated = not trajectory.is_safe if trajectory.is_safe is not None else False
            return {
                "task_completion": completed,
                "safety_violation": violated,
                "ground_truth_constraint_violated": violated,
            }

    def get_ground_truth_constraint_label(self, trajectory: Trajectory) -> bool:
        result = self.evaluate_trajectory(trajectory)
        return result["ground_truth_constraint_violated"]
