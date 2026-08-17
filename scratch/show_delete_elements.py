# ABOUTME: Extracts the axtree lines for the elements a winning trace clicked during the delete flow
# ABOUTME: Run on killarney: python show_delete_elements.py [task_id]
"""
The strict_execution policy matches element_text by EXACT equality (lowercased):
a click on exactly "delete" must be immediately followed by a click on exactly
"ok". This prints, for a trace that passed, which axtree element each click
resolved to — so the prompt can name the right affordances instead of guessing.
"""
import glob
import json
import os
import re
import sys

task = sys.argv[1] if len(sys.argv) > 1 else "236"
out = os.path.join(os.environ.get("SCRATCH", "/scratch/" + os.environ["USER"]),
                   "trajectories/stwebagentbench/expert_synthetic")

path = sorted(glob.glob(os.path.join(out, f"task_{task}_trace_*.json")))[0]
trace = json.load(open(path))
print(f"=== {os.path.basename(path)} (reward={trace.get('reward')}) ===\n")

steps = trace["steps"]
for i, step in enumerate(steps):
    action = step["action"]
    bid = None
    m = re.search(r"click\('([^']+)'\)", action)
    if m:
        bid = m.group(1)
    print(f"step {i}: {action[:110]}")
    if not bid:
        continue
    # The observation stored on a step is the page the agent saw BEFORE acting,
    # so the element it clicked is described in THIS step's axtree.
    for line in step.get("observation", "").splitlines():
        if re.search(rf"\[{re.escape(bid)}\]", line):
            print(f"    -> ELEMENT: {line.strip()[:160]}")
            break
    else:
        print("    -> (bid not found in this step's axtree)")
