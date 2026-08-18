#!/usr/bin/env python3
# ABOUTME: Audits which ST-WebAgentBench SuiteCRM tasks reference records this deployment actually seeds
# ABOUTME: Run on the login node: python audit_task_feasibility.py [--json out.json]
"""
An expert trace can only be verified when the task's target records exist. Five
tasks were already excluded for referencing empty tables; this finds the rest
before anyone writes SQL for them.

For each SuiteCRM task it pulls the quoted entity names out of the intent and
asks the database whether each one resolves to a row. Output classes:

    FEASIBLE  every quoted entity resolves — a state check can be written
    MISSING   at least one does not — needs a seed fix first, or is unusable
    NO_QUOTES no quoted entity to match on — needs a human read
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter

sys.path.insert(0, "/project/aip-s2ganapa/kunwar/icrl")

from src.trajectory_collection.stwebagentbench_state_verifier import _connect

TASKS = "/project/aip-s2ganapa/kunwar/ST-WebAgentBench/leaderboard_space/data/test.raw.json"

# Quoted strings in an intent are the record names the task acts on.
QUOTED = re.compile(r"'([^']{2,60})'")

# Values that are field contents (statuses, stages) rather than record names —
# matching them against record tables would report false misses.
FIELD_VALUES = {
    "new", "assigned", "in process", "converted", "recycled", "dead",
    "closed", "open", "open_new", "closed_closed", "assigned to me",
    "prospecting", "qualification", "needs analysis", "value proposition",
    "id. decision makers", "perception analysis", "proposal/price quote",
    "negotiation/review", "closed won", "closed lost",
    "high", "medium", "low", "urgent", "critical", "p1", "p2", "p3",
    "planned", "held", "not held", "pending input", "rejected", "duplicate",
}


def load_names(cur) -> dict[str, set[str]]:
    """Every record name in the modules this deployment seeds, lowercased."""
    out: dict[str, set[str]] = {}

    def grab(key, sql):
        try:
            cur.execute(sql)
            out[key] = {str(r[0]).strip().lower() for r in cur.fetchall() if r[0]}
        except Exception as e:
            out[key] = set()
            print(f"  (could not read {key}: {str(e)[:60]})", file=sys.stderr)

    grab("accounts", "SELECT name FROM accounts WHERE deleted=0")
    grab("opportunities", "SELECT name FROM opportunities WHERE deleted=0")
    grab("cases", "SELECT name FROM cases WHERE deleted=0")
    grab("meetings", "SELECT name FROM meetings WHERE deleted=0")
    grab("calls", "SELECT name FROM calls WHERE deleted=0")
    grab("emails", "SELECT name FROM emails WHERE deleted=0")
    grab("templates", "SELECT name FROM email_templates WHERE deleted=0")
    grab("contacts", "SELECT TRIM(CONCAT(COALESCE(first_name,''),' ',"
                     "COALESCE(last_name,''))) FROM contacts WHERE deleted=0")
    grab("leads", "SELECT TRIM(CONCAT(COALESCE(first_name,''),' ',"
                  "COALESCE(last_name,''))) FROM leads WHERE deleted=0")
    grab("users", "SELECT user_name FROM users WHERE deleted=0")
    grab("usernames", "SELECT TRIM(CONCAT(COALESCE(first_name,''),' ',"
                      "COALESCE(last_name,''))) FROM users WHERE deleted=0")
    return out


def resolves(entity: str, names: dict[str, set[str]]) -> str | None:
    """The module an entity name belongs to, or None if nothing matches."""
    e = entity.strip().lower()
    if e in FIELD_VALUES or len(e) < 3:
        return "field-value"
    for module, values in names.items():
        if e in values:
            return module
        # Surnames and partial names are common in intents ("Jim Halpert" vs
        # a contact stored with a middle name), so accept containment too.
        for v in values:
            if e in v or v in e:
                return module
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", help="write the full classification here")
    args = ap.parse_args()

    tasks = [t for t in json.load(open(TASKS))
             if t.get("sites") and "suitecrm" in str(t["sites"]).lower()]

    conn = _connect()
    cur = conn.cursor()
    names = load_names(cur)
    print("seeded records per module:")
    for k, v in sorted(names.items()):
        print(f"  {k:14s} {len(v)}")
    print()

    verdicts, rows = Counter(), []
    for t in tasks:
        intent = t.get("intent", "")
        entities = [e for e in QUOTED.findall(intent)]
        checked = {e: resolves(e, names) for e in entities}
        real = {e: m for e, m in checked.items() if m != "field-value"}

        if not entities:
            verdict = "NO_QUOTES"
        elif not real:
            verdict = "NO_QUOTES"          # only field values, nothing to locate
        elif all(m for m in real.values()):
            verdict = "FEASIBLE"
        else:
            verdict = "MISSING"

        verdicts[verdict] += 1
        rows.append({"task_id": t["task_id"], "verdict": verdict,
                     "intent": intent[:110],
                     "missing": [e for e, m in real.items() if not m],
                     "modules": sorted({m for m in real.values() if m}),
                     "n_policies": len(t.get("policies", []))})

    print("verdicts:", dict(verdicts))
    print()
    feasible = [r for r in rows if r["verdict"] == "FEASIBLE"]
    print(f"FEASIBLE task ids ({len(feasible)}):")
    print(sorted(r["task_id"] for r in feasible))
    print()
    print("module coverage among feasible:",
          Counter(m for r in feasible for m in r["modules"]).most_common())

    if args.json:
        with open(args.json, "w") as f:
            json.dump(rows, f, indent=2)
        print(f"\nfull classification → {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
