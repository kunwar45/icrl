# ABOUTME: Prints the action sequence and plan of already-verified traces, to copy what actually works
# ABOUTME: Run on killarney: python show_kept_trace.py [task_id]
import glob
import json
import os
import sys

task = sys.argv[1] if len(sys.argv) > 1 else "236"
out = os.path.join(os.environ.get("SCRATCH", "/scratch/" + os.environ["USER"]),
                   "trajectories/stwebagentbench/expert_synthetic")

for path in sorted(glob.glob(os.path.join(out, f"task_{task}_trace_*.json")))[:3]:
    trace = json.load(open(path))
    print("=" * 70)
    print(f"{os.path.basename(path)}  steps={trace['n_steps']}  "
          f"reward={trace.get('reward')}  state_verified={trace.get('state_verified')}")
    print("--- actions ---")
    for step in trace["steps"]:
        print(f"  {step['step_idx']:>2}  {step['action'][:150]}")
    print("--- plan ---")
    print("\n".join("    " + line for line in trace.get("plan", "").splitlines()[:20]))
