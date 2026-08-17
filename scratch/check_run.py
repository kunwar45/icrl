# ABOUTME: Reports the resolved config and current keep counts of the live generation run
# ABOUTME: Run on killarney: python check_run.py
import csv
import glob
import json
import os

OUT = os.path.join(os.environ.get("SCRATCH", "/scratch/" + os.environ["USER"]),
                   "trajectories/stwebagentbench/expert_synthetic")

manifest = json.load(open(os.path.join(OUT, "manifest.json")))
cfg = manifest["config"]
print("=== resolved config of the running pass ===")
print("  defer_validation :", cfg["episode"].get("defer_validation"))
print("  lean_observation :", cfg["benchmark"].get("lean_observation"))
print("  concurrency      :", cfg["generation_loop"].get("concurrency"))
print("  slow_mo_ms       :", cfg["episode"].get("slow_mo_ms"))
print("  max_revisions    :", cfg["generation_loop"].get("max_plan_revisions"))
print("  task_ids         :", cfg["benchmark"]["task_ids"])

traces = sorted(glob.glob(os.path.join(OUT, "task_*_trace_*.json")))
print(f"\n=== traces on disk: {len(traces)} ===")
by_task = {}
for path in traces:
    task = os.path.basename(path).split("_")[1]
    by_task[task] = by_task.get(task, 0) + 1
for task, n in sorted(by_task.items(), key=lambda kv: int(kv[0])):
    print(f"  task {task}: {n}")

summaries = sorted(glob.glob(os.path.join(OUT, "summary_pass_*.csv")))
if summaries:
    print(f"\n=== latest per-pass summary ({os.path.basename(summaries[-1])}) ===")
    with open(summaries[-1]) as f:
        for row in csv.DictReader(f):
            print(f"  task {row['task_id']:>4}  kept={row['kept']}  "
                  f"episodes={row['episodes_run']}  {row['outcome']}")
