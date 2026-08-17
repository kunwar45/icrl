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

    def make_env(self, task_id, max_steps=None, end_on_score=None, slow_mo_ms=None):
        self.last_max_steps = max_steps
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


def test_keep_cup_one_and_state_demands_database_proof():
    # The failure this guards: reward 1.0 from a page the agent typed into but
    # never saved (two such traces passed the benchmark evaluator in Aug 2026).
    passing = {"reward": 1.0, "violated_policies": [], "terminated": True,
               "state_verified": True}
    assert KEEP_RULES["cup_one_and_state"](passing)
    assert not KEEP_RULES["cup_one_and_state"]({**passing, "state_verified": False})
    # Never checked is not the same as verified.
    assert not KEEP_RULES["cup_one_and_state"]({**passing, "state_verified": None})
    assert not KEEP_RULES["cup_one_and_state"]({k: v for k, v in passing.items()
                                                if k != "state_verified"})
    # The cup_one half still applies.
    assert not KEEP_RULES["cup_one_and_state"]({**passing, "terminated": False})
    assert not KEEP_RULES["cup_one_and_state"]({**passing, "reward": 0.0})


def test_keep_cup_state_uses_database_not_benchmark_reward():
    # Completion is judged by the database because the benchmark's reward both
    # over-reports (unsaved typing) and under-reports (save not yet committed).
    base = {"state_verified": True, "violated_policies": [], "terminated": True,
            "finished_deliberately": True, "reward": 0.0}
    assert KEEP_RULES["cup_state"](base), "reward 0.0 must not veto database proof"
    assert not KEEP_RULES["cup_state"]({**base, "state_verified": False})
    assert not KEEP_RULES["cup_state"]({**base, "state_verified": None})
    assert not KEEP_RULES["cup_state"]({**base,
                                       "violated_policies": [{"violated": True}]})
    # A run that burned its whole step budget is a poor demonstration.
    assert not KEEP_RULES["cup_state"]({**base, "finished_deliberately": False})


def test_finished_deliberately_rejects_spinning_out_the_step_budget():
    adapter = FakeAdapter({"task_ids": [1], "script": {
        "1": [{"reward": 1.0, "terminated": True, "violations": []}]}})
    # FakeAdapter's scripted episode ends after 2 steps, below the cap.
    cfg = {"model": {"name": "m"}, "prompt": {"system": "s", "user": "u {goal}"},
           "episode": {"max_steps": 4}}
    assert run_episode(adapter, FakeClient(), cfg, 1,
                       temperature=0.0)["finished_deliberately"] is True

    # With the cap at the episode length and a non-answer final action, the
    # episode used every step it had — the spinning case seen on 2026-08-09.
    adapter2 = FakeAdapter({"task_ids": [1], "script": {
        "1": [{"reward": 1.0, "terminated": True, "violations": []}]}})
    cfg2 = {**cfg, "episode": {"max_steps": 2}}
    assert run_episode(adapter2, FakeClient(), cfg2, 1,
                       temperature=0.0)["finished_deliberately"] is False


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


def test_run_episode_state_check_is_differential():
    # A task whose goal was already satisfied before the episode cannot be
    # credited to it — otherwise a do-nothing run passes on leftover state.
    class StateAdapter(FakeAdapter):
        def __init__(self, cfg, satisfied_before):
            super().__init__(cfg)
            self._satisfied_before = satisfied_before
            self._calls = 0

        def verify_persisted_state(self, task_id):
            # First call is the pre-episode snapshot; every later call (the
            # mid-episode probe and the final settled check) sees it satisfied.
            self._calls += 1
            return (self._satisfied_before if self._calls == 1 else True), "scripted"

    cfg = {"model": {"name": "m"}, "prompt": {"system": "s", "user": "u {goal}"},
           "episode": {"max_steps": 4},
           # settle_seconds 0 keeps the test fast; production waits for the
           # final save to commit before querying.
           "verification": {"require_persisted_state": True, "settle_seconds": 0}}
    script = {"task_ids": [1], "script": {
        "1": [{"reward": 1.0, "terminated": True, "violations": []}]}}

    fresh = run_episode(StateAdapter(script, False), FakeClient(), cfg, 1, temperature=0.0)
    assert fresh["state_verified"] is True

    stale = run_episode(StateAdapter(script, True), FakeClient(), cfg, 1, temperature=0.0)
    assert stale["state_verified"] is False
    assert "ALREADY satisfied" in stale["state_detail"]


def test_run_episode_skips_state_check_when_not_requested():
    adapter = FakeAdapter({"task_ids": [1], "script": {
        "1": [{"reward": 1.0, "terminated": True, "violations": []}]}})
    cfg = {"model": {"name": "m"}, "prompt": {"system": "s", "user": "u {goal}"},
           "episode": {"max_steps": 4}}
    # FakeAdapter inherits the base verify_persisted_state, which raises — so a
    # run that does not ask for state proof must never call it.
    result = run_episode(adapter, FakeClient(), cfg, 1, temperature=0.0)
    assert result["error"] is None
    assert result["state_verified"] is None


def test_run_episode_pushes_max_steps_into_env():
    # The env must receive the episode horizon — benchmarks with an internal
    # default (ST-WebAgentBench: 20) silently cap the episode otherwise.
    adapter = FakeAdapter({"task_ids": [1], "script": {
        "1": [{"reward": 1.0, "terminated": True, "violations": []}]}})
    cfg = {"model": {"name": "m"}, "prompt": {"system": "s", "user": "u {goal}"},
           "episode": {"max_steps": 7}}
    run_episode(adapter, FakeClient(), cfg, 1, temperature=0.0)
    assert adapter.last_max_steps == 7


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


def test_page_scraped_policy_violations_are_not_binding():
    """A policy judged by scraping the agent's last page fails a compliant agent
    that navigated away (task 237 obeyed 'update the stage instead of deleting'
    and was still flagged). The database check covers the same ground, so those
    verdicts are recorded but not binding; message/sequence verdicts still are."""
    class ReportingAdapter(FakeAdapter):
        def __init__(self, cfg, report):
            super().__init__(cfg)
            self._report = report
            # unsatisfied before the episode, satisfied after — the episode's
            # own doing, which is what the differential check demands.
            self._calls = 0

        def safety_report(self, info):
            return self._report

        def verify_persisted_state(self, task_id):
            self._calls += 1
            return (self._calls > 1), "scripted"

    cfg = {"model": {"name": "m"}, "prompt": {"system": "s", "user": "u {goal}"},
           "episode": {"max_steps": 4},
           "verification": {"require_persisted_state": True, "settle_seconds": 0}}
    script = {"task_ids": [1], "script": {
        "1": [{"reward": 1.0, "terminated": True, "violations": []}]}}

    page_only = [{"violated": True, "eval_types": ["is_program_html"],
                  "policy_category": "hierarchy_adherence", "description": "d"}]
    result = run_episode(ReportingAdapter(script, page_only), FakeClient(), cfg, 1,
                         temperature=0.0)
    assert result["page_scraped_violations"] and not result["binding_violations"]
    assert KEEP_RULES["cup_state"](result), "page-scraped verdict must not veto"

    trajectory_based = [{"violated": True, "eval_types": ["is_ask_the_user"],
                         "policy_category": "user_consent", "description": "d"}]
    result = run_episode(ReportingAdapter(script, trajectory_based), FakeClient(), cfg, 1,
                         temperature=0.0)
    assert result["binding_violations"]
    assert not KEEP_RULES["cup_state"](result), "message verdict must stay binding"


def test_hints_combine_rather_than_suppressing_each_other():
    """An unsent consent message and a stuck navigation loop are independent
    problems. When the consent reminder was an elif it silently disabled
    loop-breaking for every task with a consent step."""
    seen_hints = []

    class HintCapturingAdapter(FakeAdapter):
        def step(self, env, action):
            # Never finish, so the loop runs long enough for stagnation to build.
            return {"url": "http://fake/stuck"}, 0.0, False, False, {}

        def prompt_fields(self, obs):
            fields = super().prompt_fields(obs)
            # Constant URL so the stagnation counter climbs every step.
            fields["url"] = "http://fake/stuck"
            return fields

    class RecordingClient(FakeClient):
        def __init__(self):
            super().__init__(action="click('1')")
            outer = self

            class _Completions:
                @staticmethod
                def create(**kwargs):
                    seen_hints.append(str(kwargs.get("messages", "")))

                    class _Msg: content = "click('1')"
                    class _Choice: message = _Msg()
                    class _Resp: choices = [_Choice()]
                    return _Resp()
            class _Chat: completions = _Completions()
            outer.chat = _Chat()

    adapter = HintCapturingAdapter({"task_ids": [1], "script": {
        "1": [{"reward": 0.0, "terminated": False, "violations": []}]}})
    cfg = {"model": {"name": "m"},
           "prompt": {"system": "s{hint_block}", "user": "u {goal}"},
           "episode": {"max_steps": 12}}
    run_episode(adapter, RecordingClient(), cfg, 1, temperature=0.0,
                extra_fields={"plan": "1. send_msg_to_user(\"confirm?\")"})

    combined = [h for h in seen_hints if "NOT sent yet" in h and "no progress" in h]
    assert combined, "consent reminder and stagnation warning must appear together"


def test_end_on_score_is_passed_to_the_environment():
    """The benchmark ends an episode as soon as its text-presence evaluator goes
    positive, which for seeded records happens on the first navigation. Runs must
    be able to turn that off without touching evaluation defaults."""
    class RecordingAdapter(FakeAdapter):
        def make_env(self, task_id, max_steps=None, end_on_score=None, slow_mo_ms=None):
            self.last_end_on_score = end_on_score
            return super().make_env(task_id, max_steps)

    adapter = RecordingAdapter({"task_ids": [1], "script": {
        "1": [{"reward": 1.0, "terminated": True, "violations": []}]}})
    cfg = {"model": {"name": "m"}, "prompt": {"system": "s", "user": "u {goal}"},
           "episode": {"max_steps": 4, "end_on_score": False}}
    run_episode(adapter, FakeClient(), cfg, 1, temperature=0.0)
    assert adapter.last_end_on_score is False

    # Absent from the config, the benchmark's own default must stand.
    adapter2 = RecordingAdapter({"task_ids": [1], "script": {
        "1": [{"reward": 1.0, "terminated": True, "violations": []}]}})
    run_episode(adapter2, FakeClient(),
                {**cfg, "episode": {"max_steps": 4}}, 1, temperature=0.0)
    assert adapter2.last_end_on_score is None


def test_agent_is_told_when_the_database_confirms_the_goal():
    """With end_on_score disabled the environment no longer stops the episode on
    success, so the agent must be told it succeeded — otherwise it re-attempts
    finished work, burns the budget and trips sequence policies with repeats."""
    seen = []

    class GoalAdapter(FakeAdapter):
        def __init__(self, cfg):
            super().__init__(cfg)
            self._probes = 0

        def step(self, env, action):
            return {"url": "http://fake/x"}, 1.0, False, False, {}

        def verify_persisted_state(self, task_id):
            # unsatisfied at the pre-check, satisfied from the first mid-episode probe
            self._probes += 1
            return (self._probes > 1), "scripted"

    class Recorder(FakeClient):
        def __init__(self):
            super().__init__()
            class _Completions:
                @staticmethod
                def create(**kwargs):
                    seen.append(str(kwargs.get("messages", "")))
                    class _Msg: content = "click('1')"
                    class _Choice: message = _Msg()
                    class _Resp: choices = [_Choice()]
                    return _Resp()
            class _Chat: completions = _Completions()
            self.chat = _Chat()

    cfg = {"model": {"name": "m"},
           "prompt": {"system": "s{hint_block}", "user": "u {goal}"},
           "episode": {"max_steps": 5, "end_on_score": False},
           "verification": {"require_persisted_state": True, "settle_seconds": 0}}
    result = run_episode(GoalAdapter({"task_ids": [1], "script": {
        "1": [{"reward": 1.0, "terminated": False, "violations": []}]}}),
        Recorder(), cfg, 1, temperature=0.0)

    assert any("GOAL ACHIEVED" in m for m in seen), \
        "agent must be told once the database confirms the objective"
    # Per-step scores must not be summed into a nonsense total.
    assert result["reward"] == 1.0 and result["reward_best"] == 1.0


def test_runner_ends_episode_after_grace_when_agent_ignores_goal_confirmation():
    """The agent cannot be trusted to stop when told: task 236 ran out all 30
    steps re-searching after its delete had already landed, which also tripped a
    'no intervening actions' sequence policy. The runner must end it."""
    class GoalAdapter(FakeAdapter):
        def __init__(self, cfg):
            super().__init__(cfg)
            self._calls = 0

        def step(self, env, action):
            return {"url": "http://fake/x"}, 1.0, False, False, {}

        def verify_persisted_state(self, task_id):
            self._calls += 1
            return (self._calls > 1), "scripted"   # satisfied from first probe

    cfg = {"model": {"name": "m"},
           "prompt": {"system": "s{hint_block}", "user": "u {goal}"},
           "episode": {"max_steps": 25, "end_on_score": False},
           "verification": {"require_persisted_state": True, "settle_seconds": 0}}
    result = run_episode(GoalAdapter({"task_ids": [1], "script": {
        "1": [{"reward": 1.0, "terminated": False, "violations": []}]}}),
        FakeClient(), cfg, 1, temperature=0.0)

    assert result["ended_on_goal_confirmed"] is True
    # Stopped shortly after confirmation rather than burning the whole budget.
    assert result["n_steps"] < 10, f"ran {result['n_steps']} steps"
    # And such a trace still counts as a deliberate finish.
    assert result["finished_deliberately"] is True


def test_stuck_episode_is_abandoned_instead_of_burning_the_budget():
    """SuiteCRM reassigns element ids on every re-render, so a click loop does not
    look like repeated actions. Stagnation on one URL is the reliable signal."""
    class StuckAdapter(FakeAdapter):
        def step(self, env, action):
            return {"url": "http://fake/stuck"}, 0.0, False, False, {}

        def prompt_fields(self, obs):
            fields = super().prompt_fields(obs)
            fields["url"] = "http://fake/stuck"
            return fields

    adapter = StuckAdapter({"task_ids": [1], "script": {
        "1": [{"reward": 0.0, "terminated": False, "violations": []}]}})
    cfg = {"model": {"name": "m"}, "prompt": {"system": "s{hint_block}", "user": "u"},
           "episode": {"max_steps": 30}}
    result = run_episode(adapter, FakeClient(), cfg, 1, temperature=0.0)
    assert result.get("abandoned_stuck") is True
    assert result["n_steps"] < 20, f"used {result['n_steps']} of 30 steps"


# ── Concurrency ───────────────────────────────────────────────────────────────
# Episodes are chained by the records they contend for, so rollouts of ONE task
# stay sequential (eight browsers racing to delete the same lead gives one
# delete and seven episodes flailing at a record that vanished — which
# `unsafe_binding` would happily keep), while different tasks overlap. What must
# hold either way is the bookkeeping: counts, per-task rows, one file per keep.

def test_rollouts_of_one_task_never_overlap(tmp_path, monkeypatch):
    """The default collision group is per task, so a task's own rollouts are a
    chain however high concurrency goes."""
    import threading

    in_flight = 0
    max_in_flight_per_task: dict = {}
    lock = threading.Lock()

    class ContentionTrackingAdapter(FakeAdapter):
        name = "fake"

        def make_env(self, task_id, max_steps=None, end_on_score=None, slow_mo_ms=None):
            nonlocal in_flight
            with lock:
                in_flight += 1
                max_in_flight_per_task[task_id] = max(
                    max_in_flight_per_task.get(task_id, 0), in_flight)
            import time
            time.sleep(0.02)
            with lock:
                in_flight -= 1
            return super().make_env(task_id, max_steps, end_on_score, slow_mo_ms)

    monkeypatch.setattr("src.trajectory_collection.collection_runner.get_adapter",
                        lambda bcfg: ContentionTrackingAdapter(bcfg))
    monkeypatch.setattr("src.trajectory_collection.collection_runner._build_client",
                        lambda mcfg: FakeClient())

    script = {"1": [{"reward": 1.0, "terminated": True, "violations": []}]}
    cfg = _cfg(tmp_path, mode="fixed_rollouts", keep_rule="everything",
               script=script, task_ids=[1], n_rollouts=5, concurrency=5)
    run_collection(cfg)

    assert max_in_flight_per_task[1] == 1


def test_fixed_rollouts_run_concurrently_and_keep_everything(tmp_path, monkeypatch):
    task_ids = [1, 2, 3]
    script = {str(t): [{"reward": 1.0, "terminated": True, "violations": []}]
              for t in task_ids}
    cfg = _cfg(tmp_path, mode="fixed_rollouts", keep_rule="everything",
               script=script, task_ids=task_ids, n_rollouts=4, concurrency=6)

    outcome = _run(cfg, monkeypatch)

    assert outcome["episodes"] == 12 and outcome["kept"] == 12
    assert len(list((tmp_path / "out").glob("task_*_trace_*.json"))) == 12
    # Trace numbering must not collide: 4 distinct files per task.
    for task_id in task_ids:
        assert len(list((tmp_path / "out").glob(f"task_{task_id}_trace_*.json"))) == 4


def test_concurrent_run_matches_the_serial_one(tmp_path, monkeypatch):
    """Concurrency is a throughput change, not a behaviour change."""
    task_ids = [1, 2, 3]
    script = {str(t): [{"reward": 1.0, "terminated": True, "violations": []}]
              for t in task_ids}

    def outcome_at(concurrency, out_dir):
        cfg = _cfg(tmp_path / out_dir, mode="fixed_rollouts", keep_rule="everything",
                   script=script, task_ids=task_ids, n_rollouts=3,
                   concurrency=concurrency)
        return _run(cfg, monkeypatch)

    serial = outcome_at(1, "serial")
    concurrent = outcome_at(5, "concurrent")

    assert serial["kept"] == concurrent["kept"]
    assert serial["episodes"] == concurrent["episodes"]
    assert ((tmp_path / "serial" / "out" / "summary.csv").read_text()
            == (tmp_path / "concurrent" / "out" / "summary.csv").read_text())


def test_retry_until_kept_keeps_its_early_stop_under_concurrency(tmp_path, monkeypatch):
    """A task stops at its first keep; concurrency must not turn the retry loop
    into "run every attempt anyway"."""
    task_ids = [1, 2]
    # Attempt 0 already passes, so attempts 1..4 must never run.
    script = {str(t): [{"reward": 1.0, "terminated": True, "violations": []}]
              for t in task_ids}
    cfg = _cfg(tmp_path, mode="retry_until_kept", keep_rule="cup_one",
               script=script, task_ids=task_ids, max_retries=5, concurrency=2)

    outcome = _run(cfg, monkeypatch)

    assert outcome["episodes"] == 2, "one episode per task, not five"
    assert outcome["kept"] == 2
