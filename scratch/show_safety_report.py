# ABOUTME: Prints the per-policy safety verdicts recorded on kept traces
# ABOUTME: Run on killarney: python show_safety_report.py [task_id]
import glob
import json
import os
import sys

task = sys.argv[1] if len(sys.argv) > 1 else "236"
out = os.path.join(os.environ.get("SCRATCH", "/scratch/" + os.environ["USER"]),
                   "trajectories/stwebagentbench/expert_synthetic")

for path in sorted(glob.glob(os.path.join(out, f"task_{task}_trace_*.json"))):
    trace = json.load(open(path))
    print(f"=== {os.path.basename(path)} ===")
    for report in trace.get("safety_report", []):
        print(f"  violated={str(report.get('violated')):5s} "
              f"types={report.get('eval_types')} "
              f"cat={report.get('policy_category')} "
              f"| {str(report.get('description'))[:70]}")
    print(f"  page_scraped_violations: "
          f"{len(trace.get('page_scraped_violations', []))}")
    print()
