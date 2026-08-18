#!/usr/bin/env python3
# ABOUTME: Executes every TASK_STATE_CHECKS query against the live SuiteCRM database and reports its pre-state
# ABOUTME: Run on the login node: python validate_state_checks.py
"""
A state check with a typo or a wrong column name does not fail loudly — it fails
every trace, quietly, and looks like the agent being bad at the task. This runs
each query for real and reports what it returns on the UNTOUCHED seed.

What the pre-state should be:
  * a CREATE check should return 0 (the record does not exist yet)
  * an UPDATE check should return 0 (the field does not hold the target value yet)
  * a DELETE/re-stage check may legitimately already pass

Anything that already PASSES on the seed cannot prove an agent did anything, and
anything that ERRORS is broken.
"""
from __future__ import annotations

import sys

sys.path.insert(0, "/project/aip-s2ganapa/kunwar/icrl")

from src.trajectory_collection.stwebagentbench_state_verifier import (
    TASK_CHECK_ALIASES, TASK_STATE_CHECKS, _connect)


def main() -> int:
    conn = _connect()
    cur = conn.cursor()

    broken, already_true, ok = [], [], []
    for task_id in sorted(TASK_STATE_CHECKS):
        for i, check in enumerate(TASK_STATE_CHECKS[task_id]):
            sql = check["sql"].replace("%%", "%")
            try:
                cur.execute(sql)
                count = cur.fetchone()[0]
            except Exception as e:
                broken.append((task_id, i, str(e)[:110]))
                continue

            if "equals" in check:
                passes = count == check["equals"]
                want = f"== {check['equals']}"
            else:
                passes = count >= check["at_least"]
                want = f">= {check['at_least']}"

            row = (task_id, i, count, want, passes, check["describe"][:62])
            (already_true if passes else ok).append(row)

    print(f"{len(TASK_STATE_CHECKS)} tasks, "
          f"{sum(len(v) for v in TASK_STATE_CHECKS.values())} checks, "
          f"+{len(TASK_CHECK_ALIASES)} aliased task ids\n")

    if broken:
        print("BROKEN — these never pass, whatever the agent does:")
        for task_id, i, err in broken:
            print(f"  task {task_id}[{i}]: {err}")
        print()

    print(f"correctly FALSE on the untouched seed ({len(ok)}) — these can prove work:")
    for task_id, i, count, want, _p, desc in ok:
        print(f"  task {task_id}[{i}]  got {count:>3} want {want:<6}  {desc}")

    if already_true:
        print(f"\nALREADY TRUE on the seed ({len(already_true)}) — cannot prove the "
              f"agent did anything unless the check is differential:")
        for task_id, i, count, want, _p, desc in already_true:
            print(f"  task {task_id}[{i}]  got {count:>3} want {want:<6}  {desc}")

    return 1 if broken else 0


if __name__ == "__main__":
    raise SystemExit(main())
