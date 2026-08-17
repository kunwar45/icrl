# ABOUTME: Aggregates why generation episodes were rejected, across every pass's SLURM error log.
# ABOUTME: Run on the login node: python scratch/analyze_failure_modes.py logs/slurm/icrl-generate-trajectories_4680*.err
import collections
import re
import sys

BLOCK = re.compile(r"task (\d+) revision (\d+) failed verification:")
per_task = collections.defaultdict(collections.Counter)
violations = collections.defaultdict(collections.Counter)
episodes = 0

for path in sys.argv[1:]:
    try:
        lines = open(path, errors="replace").read().splitlines()
    except OSError:
        continue
    for i, line in enumerate(lines):
        m = BLOCK.search(line)
        if not m:
            continue
        task = int(m.group(1))
        episodes += 1
        body = "\n".join(lines[i + 1:i + 26])
        reward_one = "reward: 1.0" in body
        did_persist = "TASK WAS COMPLETED SUCCESSFULLY" in body
        no_persist = "YOUR CHANGES DID NOT PERSIST" in body
        viol = re.findall(r"^\s+- \[(\w+)\] (.{0,60})", body, re.M)

        if did_persist:
            per_task[task]["completed but spun out the step budget"] += 1
        elif no_persist and viol:
            per_task[task]["no state change + policy violation"] += 1
        elif no_persist:
            per_task[task]["no state change (task not done)"] += 1
        elif viol:
            per_task[task]["policy violation only"] += 1
        elif reward_one:
            per_task[task]["other (reward 1.0)"] += 1
        else:
            per_task[task]["other"] += 1
        for category, desc in viol:
            violations[task][f"{category}: {desc.strip()}"] += 1

print(f"{episodes} rejected episodes across {len(sys.argv) - 1} logs\n")
for task in sorted(per_task):
    print(f"task {task}:")
    for reason, n in per_task[task].most_common():
        print(f"    {n:>3}x  {reason}")
    for v, n in violations[task].most_common(3):
        print(f"          violated {n}x -> {v}")
