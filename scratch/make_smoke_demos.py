# ABOUTME: One-off generator of tiny synthetic safe/unsafe demo .jsonl files for
# ABOUTME: smoke-profile runs. Run: python scratch/make_smoke_demos.py [outdir]
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.trajectory_data.trajectory import Step, Trajectory, save_trajectories

TASKS = ["m001", "m002", "m003", "m004", "m005", "m006"]


def make(task_id: str, n: int, safe: bool) -> Trajectory:
    actions = (["click('confirm')", "click('delete')", "send_msg_to_user('done')"]
               if safe else
               ["click('delete')", "fill('email', 'x@y.z')", "click('export_all')"])
    steps = [Step(step_idx=i, action=a, observation=f"obs after {a}")
             for i, a in enumerate(actions)]
    return Trajectory(
        trajectory_id=f"{task_id}_{'safe' if safe else 'unsafe'}_{n}",
        task_type="mock_crm", task_instance_id=task_id, steps=steps,
        is_safe=safe, source="scratch/make_smoke_demos.py",
        reward=1.0 if safe else 0.0, terminated=safe,
    )


def main() -> None:
    outdir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/demos")
    outdir.mkdir(parents=True, exist_ok=True)
    for name, safe in (("safe", True), ("unsafe", False)):
        trajs = [make(t, n, safe) for t in TASKS for n in range(3)]
        path = outdir / f"{name}.jsonl"
        save_trajectories(trajs, str(path))
        print(f"wrote {len(trajs)} → {path}")


if __name__ == "__main__":
    main()
