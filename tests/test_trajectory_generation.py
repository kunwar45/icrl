# ABOUTME: Unit tests for src/trajectory_generation — plan stages, failure reports, and the
# ABOUTME: generate→execute→verify→revise loop — all offline via fake adapter + fake clients.
"""
Run: pytest tests/test_trajectory_generation.py -q
"""
from __future__ import annotations

import json
from pathlib import Path

from src.trajectory_collection.benchmark_adapter import BenchmarkAdapter
from src.trajectory_generation.generation_runner import run_generation
from src.trajectory_generation.plan_generator import build_failure_report


# ── Fakes ─────────────────────────────────────────────────────────────────────

class FakeAdapter(BenchmarkAdapter):
    """Environment whose outcome depends on the CURRENT plan the executor holds.

    The fake executor echoes its plan as the action; episodes scripted in
    cfg['script'] map plan text → outcome, so a revised plan changes the result.
    """
    name = "fake"

    def make_env(self, task_id):
        return {"task_id": task_id, "step": 0, "outcome": None}

    def reset(self, env):
        return {"url": "http://fake/start"}

    def step(self, env, action):
        env["step"] += 1
        outcome = self.cfg["script"].get(action,
                                         {"reward": 0.0, "terminated": False, "violations": []})
        done = env["step"] >= 2
        reward = outcome["reward"] if done else 0.0
        info = {"violations": outcome["violations"]} if done else {}
        terminated = done and outcome["terminated"]
        truncated = done and not outcome["terminated"]
        return {"url": f"http://fake/{env['step']}"}, reward, terminated, truncated, info

    def prompt_fields(self, obs):
        return {"goal": "delete the record", "url": obs["url"], "axtree": "tree",
                "chat_history": "", "policies_block": "confirm before delete",
                "action_space": "click/fill/answer"}

    def parse_action(self, llm_output):
        return llm_output.strip()

    def safety_report(self, info):
        return [{"policy_id": c, "policy_category": c, "description": c,
                 "violated": True, "reason": "scripted"}
                for c in info.get("violations", [])]

    def task_ids(self):
        return list(self.cfg["task_ids"])

    def task_metadata(self, task_id):
        # Override: no env boot needed for the fake.
        return {"goal": "delete the record", "policies_block": "confirm before delete",
                "action_space": "click/fill/answer"}


class ScriptedPlanner:
    """Returns queued responses in order: propose, refine, revise, revise, ..."""
    def __init__(self, responses):
        it = iter(responses)
        class _Completions:
            @staticmethod
            def create(**kwargs):
                text = next(it)
                class _Msg: content = text
                class _Choice: message = _Msg()
                class _Resp: choices = [_Choice()]
                return _Resp()
        class _Chat: completions = _Completions()
        self.chat = _Chat()


class PlanEchoExecutor:
    """Fake policy model: acts by echoing the plan it was given in the system prompt."""
    def __init__(self):
        class _Completions:
            @staticmethod
            def create(messages=None, **kwargs):
                system = messages[0]["content"]
                # the plan is injected via {plan}; our templates put it on its own line
                plan_line = [l for l in system.splitlines() if l.startswith("PLAN:")][0]
                text = plan_line.removeprefix("PLAN:")
                class _Msg: content = text
                class _Choice: message = _Msg()
                class _Resp: choices = [_Choice()]
                return _Resp()
        class _Chat: completions = _Completions()
        self.chat = _Chat()


def _cfg(tmp_path: Path, script, max_revisions=2, task_ids=(1,), diversity=None):
    return {
        "generation": "fake_generation_test",
        "benchmark": {"name": "fake", "task_ids": list(task_ids), "script": script},
        **({"diversity": diversity} if diversity else {}),
        "models": {
            "planner": {"backend": "fake", "name": "planner-model", "max_tokens": 256,
                        "temperature": 0.7},
            "executor": {"backend": "fake", "name": "executor-model", "max_tokens": 64,
                         "temperature": 0.0},
        },
        "episode": {"max_steps": 4},
        "generation_loop": {"max_plan_revisions": max_revisions},
        "keep": {"rule": "cup_one", "set": "expert_synthetic"},
        "output": {"dir": str(tmp_path / "out")},
        "prompts": {
            "propose_plan": {"system": "s", "user": "{goal} {policies_block}"},
            "refine_plan": {"system": "s", "user": "{plan}\n{previous_plans}"},
            "diversify_plan": {"system": "s", "user": "{plan}\n{previous_plans}"},
            "revise_plan": {"system": "s", "user": "{plan} {failure_report}"},
            "execute": {"system": "PLAN:{plan}\n{policies_block}{hint_block}",
                        "user": "{goal} {url} {axtree} {actions_so_far} {action_space}"},
        },
    }


def _run(cfg, planner, monkeypatch):
    monkeypatch.setattr("src.trajectory_generation.generation_runner.get_adapter",
                        lambda bcfg: FakeAdapter(bcfg))
    clients = {"planner-model": planner, "executor-model": PlanEchoExecutor()}
    monkeypatch.setattr("src.trajectory_generation.generation_runner._build_client",
                        lambda mcfg: clients[mcfg["name"]])
    return run_generation(cfg)


# ── Failure report ────────────────────────────────────────────────────────────

def test_failure_report_names_violations_and_truncation():
    report = build_failure_report({
        "reward": 0.0, "terminated": False, "n_steps": 30,
        "violated_policies": [{"policy_category": "user_consent",
                               "description": "confirm first", "reason": "no msg"}],
        "steps": [{"action": "click('9')"}], "error": None,
    })
    assert "user_consent" in report
    assert "ran out of steps" in report
    assert "click('9')" in report


# ── The loop ──────────────────────────────────────────────────────────────────

def test_verified_plan_saved_with_provenance(tmp_path, monkeypatch):
    # refine output "good_plan" → executor echoes it → scripted env passes it
    script = {"good_plan": {"reward": 1.0, "terminated": True, "violations": []}}
    planner = ScriptedPlanner(["draft_plan", "good_plan"])
    outcome = _run(_cfg(tmp_path, script), planner, monkeypatch)

    assert outcome["kept"] == 1 and outcome["episodes"] == 1
    trace = json.loads((tmp_path / "out" / "task_1_trace_0.json").read_text())
    assert trace["set"] == "expert_synthetic"
    assert trace["pipeline"] == "trajectory_generation"
    assert trace["plan"] == "good_plan"
    assert trace["plan_revisions"] == 0
    assert trace["cup"] is True
    assert trace["model"] == "executor-model"
    assert trace["planner_model"] == "planner-model"


def test_failed_plan_gets_revised_then_passes(tmp_path, monkeypatch):
    script = {
        "bad_plan": {"reward": 0.0, "terminated": False, "violations": ["user_consent"]},
        "fixed_plan": {"reward": 1.0, "terminated": True, "violations": []},
    }
    # propose → refine → (execution fails) → revise → "fixed_plan"
    planner = ScriptedPlanner(["draft", "bad_plan", "fixed_plan"])
    outcome = _run(_cfg(tmp_path, script), planner, monkeypatch)

    assert outcome["kept"] == 1 and outcome["episodes"] == 2
    trace = json.loads((tmp_path / "out" / "task_1_trace_0.json").read_text())
    assert trace["plan"] == "fixed_plan"
    assert trace["plan_revisions"] == 1


def test_similar_plan_triggers_diversify_pass(tmp_path, monkeypatch):
    # Task 1 succeeds with "plan_alpha_for_task"; task 2's refined plan is
    # near-identical, so the diversity gate must fire and use the diversified one.
    good_a = "plan_alpha_for_task steps one two three four five"
    near_dup = "plan_alpha_for_task steps one two three four SIX"
    distinct = "totally_different_route_via_search"
    script = {
        good_a: {"reward": 1.0, "terminated": True, "violations": []},
        distinct: {"reward": 1.0, "terminated": True, "violations": []},
    }
    # task1: propose, refine → good_a (kept, enters history)
    # task2: propose, refine → near_dup (similarity fires) → diversify → distinct
    planner = ScriptedPlanner(["draft1", good_a, "draft2", near_dup, distinct])
    cfg = _cfg(tmp_path, script, task_ids=[1, 2],
               diversity={"enabled": True, "max_similarity": 0.85, "history": 8})
    outcome = _run(cfg, planner, monkeypatch)

    assert outcome["kept"] == 2
    trace2 = json.loads((tmp_path / "out" / "task_2_trace_0.json").read_text())
    assert trace2["plan"] == distinct

    summary = (tmp_path / "out" / "summary.csv").read_text()
    assert "plan_similarity" in summary.splitlines()[0]


def test_only_successful_plans_enter_history(tmp_path, monkeypatch):
    # Task 1 fails all attempts with "failing_plan"; task 2's refined plan is
    # near-identical to it — the gate must NOT fire, because failed plans never
    # enter the diversity context.
    failing = "shared_wording_plan steps one two three four five"
    near_dup = "shared_wording_plan steps one two three four SIX"
    script = {near_dup: {"reward": 1.0, "terminated": True, "violations": []}}
    # task1: propose, refine → failing; 2 executions fail; 1 revise → failing again
    # task2: propose, refine → near_dup — no diversify call must happen
    planner = ScriptedPlanner(["d1", failing, failing, "d2", near_dup])
    cfg = _cfg(tmp_path, script, max_revisions=1, task_ids=[1, 2],
               diversity={"enabled": True, "max_similarity": 0.85, "history": 8})
    outcome = _run(cfg, planner, monkeypatch)

    assert outcome["kept"] == 1  # only task 2
    trace2 = json.loads((tmp_path / "out" / "task_2_trace_0.json").read_text())
    assert trace2["plan"] == near_dup  # used as-is, no diversification


def test_exhausted_revisions_saves_nothing(tmp_path, monkeypatch):
    script = {}  # every plan fails
    planner = ScriptedPlanner(["draft", "bad1", "bad2", "bad3"])
    outcome = _run(_cfg(tmp_path, script, max_revisions=2), planner, monkeypatch)

    assert outcome["kept"] == 0
    assert outcome["episodes"] == 3  # initial + 2 revisions
    assert not list((tmp_path / "out").glob("task_*_trace_*.json"))
    summary = (tmp_path / "out" / "summary.csv").read_text()
    assert "exhausted revisions" in summary
