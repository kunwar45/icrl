#!/usr/bin/env python3
# ABOUTME: Finds ST-WebAgentBench SuiteCRM tasks whose end state is a trivially checkable CRUD operation
# ABOUTME: Run on the login node: python audit_crud_tasks.py
"""
Which tasks can cheaply get a database state check?

A CREATE task is verifiable only when its target does NOT already exist in the
seed — otherwise "did it get created" has no honest answer, which is exactly why
242 and 243 were excluded. An UPDATE or DELETE task is verifiable only when its
target DOES exist.

This checks both directions against the live database and prints the tasks that
are ready for a check, grouped by the SQL pattern they need.
"""
from __future__ import annotations

import collections
import json
import re
import sys

sys.path.insert(0, "/project/aip-s2ganapa/kunwar/icrl")

from src.trajectory_collection.stwebagentbench_state_verifier import _connect

TASKS = "/project/aip-s2ganapa/kunwar/ST-WebAgentBench/leaderboard_space/data/test.raw.json"

# Which module a "create a new <thing>" / "go to the <thing>" intent targets.
MODULE_WORDS = {
    "account": "accounts", "contact": "contacts", "lead": "leads",
    "opportunity": "opportunities", "case": "cases", "meeting": "meetings",
    "call": "calls", "task": "tasks", "email": "emails", "note": "notes",
    "document": "documents",
}

NAME_COLUMN = {
    "accounts": "name", "opportunities": "name", "cases": "name",
    "meetings": "name", "calls": "name", "emails": "name", "tasks": "name",
    "notes": "name", "documents": "document_name",
    "contacts": "CONCAT(COALESCE(first_name,''),' ',COALESCE(last_name,''))",
    "leads": "CONCAT(COALESCE(first_name,''),' ',COALESCE(last_name,''))",
}

QUOTED = re.compile(r"'([^']{2,60})'")


def classify(intent: str) -> tuple[str, str | None]:
    """(verb, module) for an intent, or ('other', None)."""
    low = intent.lower()
    module = None
    for word, table in MODULE_WORDS.items():
        if re.search(rf"\b{word}s?\b", low):
            module = table
            break
    if low.startswith("create"):
        return "create", module
    if low.startswith(("go to", "update", "change", "set", "edit")):
        return "update", module
    if low.startswith(("delete", "remove")):
        return "delete", module
    return "other", module


def exists(cur, module: str, value: str) -> bool | None:
    col = NAME_COLUMN.get(module)
    if not col:
        return None
    try:
        cur.execute(
            f"SELECT COUNT(*) FROM {module} WHERE deleted=0 AND TRIM({col})=%s",
            (value.strip(),))
        return cur.fetchone()[0] > 0
    except Exception:
        return None


def main() -> int:
    tasks = [t for t in json.load(open(TASKS))
             if t.get("sites") and "suitecrm" in str(t["sites"]).lower()]
    fam = collections.defaultdict(list)
    for t in tasks:
        fam[t["intent"].strip()].append(t["task_id"])

    conn = _connect()
    cur = conn.cursor()

    ready, blocked, skipped = [], [], 0
    for intent, ids in sorted(fam.items(), key=lambda kv: min(kv[1])):
        verb, module = classify(intent)
        quoted = QUOTED.findall(intent)
        if verb == "other" or not module or not quoted:
            skipped += 1
            continue
        target = quoted[0]
        present = exists(cur, module, target)
        if present is None:
            skipped += 1
            continue

        # A create needs its target ABSENT; an update/delete needs it PRESENT.
        ok = (not present) if verb == "create" else present
        row = {"ids": sorted(ids), "verb": verb, "module": module,
               "target": target, "seeded": present, "intent": intent[:95],
               "n_policies": max(len(t.get("policies", [])) for t in tasks
                                 if t["intent"].strip() == intent)}
        (ready if ok else blocked).append(row)

    print(f"READY for a state check: {len(ready)} intents "
          f"({sum(len(r['ids']) for r in ready)} task ids)")
    by_pattern = collections.Counter((r["verb"], r["module"]) for r in ready)
    for (verb, module), n in by_pattern.most_common():
        print(f"   {verb:7s} {module:14s} {n} intent(s)")
    print()
    for r in ready:
        print(f"  {str(r['ids']):18s} {r['verb']:7s} {r['module']:14s} "
              f"pol={r['n_policies']:2d}  {r['intent']}")

    print(f"\nBLOCKED (create whose target already exists, or update/delete "
          f"whose target is missing): {len(blocked)} intents")
    for r in blocked[:12]:
        why = "target already seeded" if r["verb"] == "create" else "target not seeded"
        print(f"  {str(r['ids']):18s} {r['verb']:7s} {why:22s} {r['intent'][:70]}")
    print(f"\nnot classifiable from the intent alone: {skipped}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
