# ABOUTME: Lists every button/link in the confirm-dialog page state of a kept trace
# ABOUTME: Run on killarney: python show_dialog_buttons.py [task_id]
"""
The strict_execution policy demands a click on element_text exactly "ok".
This prints every clickable element the agent could see at the confirmation
step, so we can tell whether such an element exists in THIS deployment or the
policy is unsatisfiable here.
"""
import glob
import json
import os
import re
import sys

task = sys.argv[1] if len(sys.argv) > 1 else "236"
out = os.path.join(os.environ.get("SCRATCH", "/scratch/" + os.environ["USER"]),
                   "trajectories/stwebagentbench/expert_synthetic")

for path in sorted(glob.glob(os.path.join(out, f"task_{task}_trace_*.json"))):
    trace = json.load(open(path))
    steps = trace["steps"]
    # The confirmation click is the last click of the episode.
    idx = max((i for i, s in enumerate(steps) if s["action"].startswith("click")),
              default=None)
    if idx is None:
        continue
    print(f"=== {os.path.basename(path)}  (final click: {steps[idx]['action']}) ===")
    observation = steps[idx].get("observation", "")
    hits = [line.strip() for line in observation.splitlines()
            if re.search(r"\b(button|link)\b", line)
            and re.search(r"proceed|ok|cancel|confirm|yes|delete", line, re.I)]
    for line in hits[:12]:
        print("   ", line[:150])
    if not hits:
        print("    (no matching clickable elements found in this observation)")
    print()
