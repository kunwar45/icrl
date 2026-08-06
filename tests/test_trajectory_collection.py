# ABOUTME: Unit tests for src/trajectory_collection — keep rules, temperature schedule, episode
# ABOUTME: loop, and the collection runner — all offline via a fake adapter + fake client.
"""
No browsergym, no network: the BenchmarkAdapter seam is exercised with a
scripted fake so the engine's behaviour (retry/keep/save/resume) is testable on
a laptop. Run: pytest tests/test_data_collection.py -q
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.trajectory_collection.benchmark_adapter import BenchmarkAdapter, get_adapter
from src.trajectory_collection.collection_runner import (KEEP_RULES, _next_trace_n,
                                                   _temperature, run_collection)
from src.trajectory_collection.episode_runner import run_episode


# ── Fakes ─────────────────────────────────────────────────────────────────────

class FakeAdapter(BenchmarkAdapter):
    """Scripted environment: episodes defined per task in cfg['script'].

    Each scripted episode: {"reward": float, "terminated": bool,
    "violations": [category, ...]} — consumed one per attempt, last repeats.
    """
    name = "fake"

    def __init__(self, benchmark_cfg):
        super().__init__(benchmark_cfg)
        self._attempts: dict = {}

    def _episode(self, task_id):
        script = self.cfg["script"][str(task_id)]
        i = min(self._attempts.get(task_id, 0), len(script) - 1)
        self._attempts[task_id] = self._attempts.get(task_id, 0) + 1
        return script[i]

    def make_env(self, task_id):
        return {"task_id": task_id, "episode": self._episode(task_id), "step": 0}

    def reset(self, env):
        return {"url": "http://fake/start"}

    def step(self, env, action):
        env["step"] += 1
        ep = env["episode"]
        done = env["step"] >= 2
        reward = ep["reward"] if done else 0.0
        info = {"violations": ep["violations"]} if done else {}
        return {"url": f"http://fake/{env['step']}"}, reward, done and ep["terminated"], done and not ep["terminated"], info

    def prompt_fields(self, obs):
        return {"goal": "g", "url": obs["url"], "axtree": "tree",
                "chat_history": "", "policies_block": "P", "action_space": "A"}

    def parse_action(self, llm_output):
        return llm_output if llm_output.endswith(")") else None

    def safety_report(self, info):
        return [{"policy_id": c, "policy_category": c, "description": c,
                 "violated": True, "reason": "scripted"}
                for c in info.get("violations", [])]

    def task_ids(self):
        return list(self.cfg["task_ids"])


class FakeClient:
    """Always answers with a fixed action."""
    def __init__(self, action="click('1')"):
        class _Completions:
            @staticmethod
            def create(**kwargs):
                class _Msg: content = action
                class _Choice: message = _Msg()
                class _Resp: choices = [_Choice()]
                return _Resp()
        class _Chat: completions = _Completions()
        self.chat = _Chat()


def _cfg(tmp_path: Path, *, mode, keep_rule, script, task_ids, **episode_extra):
    return {
        "collection": "fake_test",
        "benchmark": {"name": "fake", "task_ids": task_ids, "script": script},
        "model": {"backend": "fake", "name": "fake-model", "max_tokens": 64},
        "episode": {"mode": mode, "max_steps": 4,
                    "temperature": {"first": 0.0, "base": 0.2, "step": 0.15, "max": 0.8},
                    **episode_extra},
        "keep": {"rule": keep_rule, "set": "expert"},
        "prompt": {"system": "s {policies_block}{hint_block}", "user": "u {goal}"},
        "output": {"dir": str(tmp_path / "out")},
    }


def _run(cfg, monkeypatch):
    monkeypatch.setattr("src.trajectory_collection.collection_runner.get_adapter",
                        lambda bcfg: FakeAdapter(bcfg))
    monkeypatch.setattr("src.trajectory_collection.collection_runner._build_client",
                        lambda mcfg: FakeClient())
    return run_collection(cfg)


# ── Pure helpers ──────────────────────────────────────────────────────────────

def test_temperature_schedule_greedy_then_ramps():
    sched = {"first": 0.0, "base": 0.2, "step": 0.15, "max": 0.8}
    temps = [_temperature(a, sched) for a in range(6)]
    assert temps[0] == 0.0
    assert temps[1] == pytest.approx(0.35)
    assert temps[-1] == 0.8
    assert temps == sorted(temps)  # never cools back down


def test_keep_cup_one_requires_termination():
    base = {"reward": 1.0, "violated_policies": [], "terminated": True}
    assert KEEP_RULES["cup_one"](base)
    assert not KEEP_RULES["cup_one"]({**base, "terminated": False})
    assert not KEEP_RULES["cup_one"]({**base, "reward": 0.0})
    assert not KEEP_RULES["cup_one"]({**base, "violated_policies": [{"violated": True}]})


def test_keep_any_violation():
    assert KEEP_RULES["any_violation"](
        {"reward": 0.0, "violated_policies": [{"violated": True}], "terminated": False})
    assert not KEEP_RULES["any_violation"](
        {"reward": 1.0, "violated_policies": [], "terminated": True})


def test_next_trace_n_resumes_from_existing(tmp_path):
    (tmp_path / "task_235_trace_0.json").write_text("{}")
    (tmp_path / "task_235_trace_3.json").write_text("{}")
    assert _next_trace_n(tmp_path, 235) == 4
    assert _next_trace_n(tmp_path, 236) == 0


# ── Adapter registry ──────────────────────────────────────────────────────────

def test_get_adapter_rejects_unknown_benchmark():
    with pytest.raises(ModuleNotFoundError):
        get_adapter({"name": "no_such_benchmark"})


def test_get_adapter_requires_name():
    with pytest.raises(ValueError):
        get_adapter({})


# ── Episode runner ────────────────────────────────────────────────────────────

def test_run_episode_records_steps_and_report():
    adapter = FakeAdapter({"task_ids": [1], "script": {
        "1": [{"reward": 1.0, "terminated": True, "violations": []}]}})
    cfg = {"model": {"name": "m"}, "prompt": {"system": "s {policies_block}", "user": "u {goal}"},
           "episode": {"max_steps": 4}}
    result = run_episode(adapter, FakeClient(), cfg, 1, temperature=0.0)
    assert result["error"] is None
    assert result["reward"] == 1.0
    assert result["terminated"]
    assert result["n_steps"] == 2
    assert result["steps"][0]["action"] == "click('1')"


def test_run_episode_survives_unparsable_output():
    adapter = FakeAdapter({"task_ids": [1], "script": {
        "1": [{"reward": 0.0, "terminated": False, "violations": []}]}})
    cfg = {"model": {"name": "m"}, "prompt": {"system": "s", "user": "u"},
           "episode": {"max_steps": 3}}
    result = run_episode(adapter, FakeClient(action="no action here"), cfg, 1, 0.0)
    assert result["error"] is None
    assert all(s["action"] == "noop()" for s in result["steps"])


# ── Collection runner ─────────────────────────────────────────────────────────

def test_retry_until_kept_stops_at_first_success(tmp_path, monkeypatch):
    script = {"7": [
        {"reward": 0.0, "terminated": False, "violations": []},   # attempt 1 fails
        {"reward": 1.0, "terminated": True, "violations": []},    # attempt 2 succeeds
        {"reward": 1.0, "terminated": True, "violations": []},    # never reached
    ]}
    cfg = _cfg(tmp_path, mode="retry_until_kept", keep_rule="cup_one",
               script=script, task_ids=[7], max_retries=3)
    outcome = _run(cfg, monkeypatch)
    assert outcome["kept"] == 1
    assert outcome["episodes"] == 2  # stopped after success, not all 3

    trace = json.loads((tmp_path / "out" / "task_7_trace_0.json").read_text())
    assert trace["cup"] is True
    assert trace["set"] == "expert"
    assert trace["collection"] == "fake_test"

    manifest = json.loads((tmp_path / "out" / "manifest.json").read_text())
    assert manifest["kept"] == 1 and "finished" in manifest


def test_fixed_rollouts_keeps_every_match(tmp_path, monkeypatch):
    script = {"9": [
        {"reward": 0.0, "terminated": False, "violations": ["user_consent"]},
        {"reward": 0.0, "terminated": False, "violations": []},
        {"reward": 0.0, "terminated": False, "violations": ["boundary_scope"]},
    ]}
    cfg = _cfg(tmp_path, mode="fixed_rollouts", keep_rule="any_violation",
               script=script, task_ids=[9], n_rollouts=3)
    cfg["keep"]["set"] = "unsafe"
    outcome = _run(cfg, monkeypatch)
    assert outcome["episodes"] == 3
    assert outcome["kept"] == 2  # the two violating episodes, not the clean one

    summary = (tmp_path / "out" / "summary.csv").read_text()
    assert "boundary_scope" in summary and "user_consent" in summary


def test_nothing_kept_reports_zero(tmp_path, monkeypatch):
    script = {"5": [{"reward": 0.0, "terminated": False, "violations": []}]}
    cfg = _cfg(tmp_path, mode="retry_until_kept", keep_rule="cup_one",
               script=script, task_ids=[5], max_retries=2)
    outcome = _run(cfg, monkeypatch)
    assert outcome["kept"] == 0
    assert not list((tmp_path / "out").glob("task_*_trace_*.json"))
