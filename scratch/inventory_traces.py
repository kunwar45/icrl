# ABOUTME: Inventories every trajectory on the cluster with the signals that decide whether it is training-worthy.
# ABOUTME: Run on the login node: python scratch/inventory_traces.py
import glob
import json
import os

DIRS = [
    "/scratch/kunwar/trajectories/stwebagentbench/expert_synthetic",
    "/scratch/kunwar/trajectories/stwebagentbench/_quarantine_false_passes",
    "/project/aip-s2ganapa/kunwar/trajectories/safe",
]

for directory in DIRS:
    files = sorted(glob.glob(os.path.join(directory, "task_*.json")))
    print(f"=== {directory}  ({len(files)} traces)")
    for path in files:
        trace = json.load(open(path))
        steps = trace.get("steps") or []
        actions = [s.get("action", "") for s in steps]
        consent = sum(1 for a in actions if a.startswith("send_msg_to_user"))
        violated = [p for p in (trace.get("safety_report") or []) if p.get("violated")]
        print(f"  {os.path.basename(path)}")
        print(f"      steps={len(steps)} reward={trace.get('reward')} "
              f"state_verified={trace.get('state_verified')} "
              f"violations={len(violated)} consent_msgs={consent} "
              f"pipeline={trace.get('pipeline', '(collection)')}")
        for a in actions[:8]:
            print(f"        {a[:100]}")
    print()
