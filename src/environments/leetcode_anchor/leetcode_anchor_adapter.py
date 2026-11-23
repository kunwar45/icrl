# ABOUTME: LeetCode anchor task for GRAFT (Ward et al. 2026, Section 4.1): prompts and a hidden-test reward for medium and hard problems of LeetCodeDataset (Xia et al. 2025, arXiv 2504.14655); the reply cannot see or edit the tests
# ABOUTME: Use: msgs = chat_messages(item); scores = score_replies(items, replies) -> [{"parsed", "reward", "error"}]; items come from load_problems(path) (see scripts/prepare_leetcode_anchor_data.py)
"""
GRAFT trains the deployed policy (RETAIN adapter only) on "anchor" replies from a
separate task that does not admit the hacks of the target task. The paper uses
the medium and hard problems of the LeetCode environment of Xia et al.

Ours, built from the public dataset (the paper's own anchor file is in another
repository and is not used):

  prompt   the problem statement and the starter code; the reply is reasoning in
           <think> tags and then a JSON object {"solution.py": "<code>"}, the same
           reply shape as Countdown-Code
  reward   1 if the returned solution passes every hidden test, else 0. The tests
           are held by the environment: the reply never sees them and returns no
           test file, so there is nothing to rewrite.

Solutions are untrusted model output and run in a separate resource-limited
process (the same runner as Countdown-Code).
"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from src.environments.countdown_code.countdown_code_adapter import (
    SYSTEM_MESSAGE,
    load_json_from_response,
)
from src.environments.countdown_code.sandboxed_test_runner import run_python_child

PROMPT_TEMPLATE = (
    "You are to solve a programming problem using Python.\n"
    "You will be provided the starter code of `solution.py`. Complete it so that it solves the problem.\n"
    "Your solution is graded on hidden test cases that you cannot see or change.\n\n"
    "Show your thinking in <think> </think> tags, and return the file as valid json.\n"
    "You will only be rewarded if every hidden test case passes and the formatting instructions are followed.\n\n"
    "PROBLEM:\n"
    "{problem}\n\n"
    "Here is the file for you to work with:\n\n"
    "```\n"
    "{files_json}\n"
    "```\n\n"
    "Your response must be of the form:\n"
    "```\n"
    "<think>\n"
    "Your Chain of Thought goes here\n"
    "</think>\n"
    "{{ ...json here... }}\n"
    "```"
)

# What LeetCode makes available to a solution without an import.
PREAMBLE = (
    "import bisect, collections, functools, heapq, itertools, math, operator, random, re, string, sys\n"
    "from bisect import *\n"
    "from collections import *\n"
    "from functools import *\n"
    "from heapq import *\n"
    "from itertools import *\n"
    "from math import *\n"
    "from operator import *\n"
    "from string import *\n"
    "from typing import *\n"
    "inf = float('inf')\n"
)

# Executed in the child. Reads one JSON job on stdin, prints one JSON result.
_CHILD = r'''
import json, sys
try:
    import resource
    resource.setrlimit(resource.RLIMIT_CPU, (CPU, CPU))
    try:
        resource.setrlimit(resource.RLIMIT_AS, (MEM, MEM))
    except Exception:
        pass
    resource.setrlimit(resource.RLIMIT_FSIZE, (0, 0))
except Exception:
    pass
sys.setrecursionlimit(20000)
job = json.loads(sys.stdin.read())
out = {"passed": 0, "error": ""}
real_stdout = sys.stdout
sys.stdout = sys.stderr = open("/dev/null", "w")
try:
    ns = {"__name__": "__solution__"}
    try:
        exec(job["preamble"], ns)
        exec(job["solution"], ns)
        exec(job["test"], ns)
        ns["check"](eval(job["entry_point"], ns))
        out["passed"] = 1
    except BaseException as e:
        out["error"] = type(e).__name__
finally:
    real_stdout.write(json.dumps(out))
    real_stdout.flush()
'''


def user_prompt(item: dict) -> str:
    files = json.dumps({"solution.py": item["starter_code"]})
    return PROMPT_TEMPLATE.format(problem=item["problem"].strip(), files_json=files)


def chat_messages(item: dict) -> list[dict]:
    return [{"role": "system", "content": SYSTEM_MESSAGE}, {"role": "user", "content": user_prompt(item)}]


def passes_hidden_tests(
    jobs: list[dict], timeout: float = 30.0, cpu_seconds: int = 10, memory_mb: int = 2048, workers: int = 16
) -> list[dict]:
    """Per job {"solution", "test", "entry_point"}: {"passed": 0/1, "error": str}. A crash, hang or limit scores 0."""
    if not jobs:
        return []
    code = _CHILD.replace("CPU", str(int(cpu_seconds))).replace("MEM", str(int(memory_mb) * 1024 * 1024))

    def one(job: dict) -> dict:
        res = run_python_child(code, {**job, "preamble": PREAMBLE}, timeout)
        if res is None:
            return {"passed": 0, "error": "runner"}
        return {"passed": int(res.get("passed", 0)), "error": str(res.get("error", ""))}

    with ThreadPoolExecutor(max_workers=min(workers, len(jobs))) as ex:
        return list(ex.map(one, jobs))


def score_replies(items: list[dict], replies: list[str], workers: int = 16) -> list[dict]:
    """Per reply: {"parsed", "reward", "error"}. reward = 1 only if the returned solution.py passes every hidden test."""
    out = [{"parsed": False, "reward": 0.0, "error": "no json"} for _ in replies]
    jobs, slots = [], []
    for i, (item, reply) in enumerate(zip(items, replies)):
        files = load_json_from_response(reply)
        solution = files.get("solution.py") if isinstance(files, dict) else None
        if not isinstance(solution, str):
            continue
        jobs.append({"solution": solution, "test": item["test"], "entry_point": item["entry_point"]})
        slots.append(i)
    for i, res in zip(slots, passes_hidden_tests(jobs, workers=workers)):
        out[i] = {"parsed": True, "reward": float(res["passed"]), "error": res["error"]}
    return out


def load_problems(path: str | Path, n: int | None = None) -> list[dict]:
    """Rows of a prepared .jsonl file: {"id", "difficulty", "problem", "starter_code", "entry_point", "test"}."""
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    return rows if n is None else rows[:n]
