"""
Thin wrapper around ST-WebAgentBench.
ST-WebAgentBench repo: https://github.com/segev-shlomov/ST-WebAgentBench
Install: pip install st-webagentbench  (or clone + pip install -e .)
"""
from typing import List, Dict, Optional
from src.data.trajectory import Trajectory, Step


TASK_TYPES = [
    "delete_record",
    "expose_private_field",
    "irreversible_submit",
    "skip_confirmation",
    "bulk_destructive_action",
]


class STWebAgentBench:
    def __init__(self, benchmark_root: str):
        self.root = benchmark_root
        self._tasks: Optional[Dict] = None

    def load_tasks(self) -> Dict:
        # TODO (engineer): load from benchmark_root/tasks/*.json
        # Each task JSON: task_id, task_type, constraint_description, ground_truth_label
        raise NotImplementedError

    def get_tasks_by_type(self, task_type: str) -> List[Dict]:
        if self._tasks is None:
            self._tasks = self.load_tasks()
        return [t for t in self._tasks.values() if t["task_type"] == task_type]

    def evaluate_trajectory(self, trajectory: Trajectory) -> Dict:
        # TODO (engineer): call the benchmark's evaluator here.
        # Returns: task_completion (bool), safety_violation (bool),
        #          ground_truth_constraint_violated (bool)
        raise NotImplementedError

    def get_ground_truth_constraint_label(self, trajectory: Trajectory) -> bool:
        result = self.evaluate_trajectory(trajectory)
        return result["ground_truth_constraint_violated"]
