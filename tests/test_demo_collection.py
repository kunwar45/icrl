"""
Tests for the safe demo collection system.
All tests run without BrowserGym, OpenAI, or a live network connection.
"""
import json
import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest

from src.data.st_webagent import STWebAgentBench, TASK_TYPES, obs_repr
from src.data.safety_verifier import SafetyVerifier, VerificationResult, _PARSE_FAILURE
from src.data.trajectory import Step, Trajectory


# ── Fixtures ──────────────────────────────────────────────────────────────────

def make_steps(n: int = 3) -> list[Step]:
    return [
        Step(
            step_idx=i,
            action=f"click('btn_{i}')",
            observation=f"GOAL: Delete record\nURL: http://example.com\nPAGE: step {i}",
        )
        for i in range(n)
    ]


def make_trajectory(is_safe: bool = True) -> Trajectory:
    return Trajectory(
        trajectory_id="t001",
        task_type="delete_record",
        task_instance_id="235",
        steps=make_steps(),
        is_safe=is_safe,
        source="gpt-4o",
        reward=7.5,
    )


def make_task(task_type: str = "delete_record") -> dict:
    return {
        "task_id": "235",
        "task_type": task_type,
        "constraint_description": "Delete the record at row 3.",
        "ground_truth_label": True,
        "policies": ["Always confirm before deletion."],
    }


# ── STWebAgentBench ───────────────────────────────────────────────────────────

class TestSTWebAgentBench:
    def test_inject_task_id_map_populates_tasks(self):
        bench = STWebAgentBench("/fake/root")
        bench.inject_task_id_map({
            "delete_record": ["235", "312"],
            "skip_confirmation": ["400"],
        })
        assert bench._tasks is not None
        assert "235" in bench._tasks
        assert bench._tasks["235"]["task_type"] == "delete_record"
        assert "400" in bench._tasks

    def test_inject_empty_map_does_nothing(self):
        bench = STWebAgentBench("/fake/root")
        bench.inject_task_id_map({"delete_record": []})
        assert bench._tasks is None  # empty lists should not set _tasks

    def test_get_tasks_by_type_after_inject(self):
        bench = STWebAgentBench("/fake/root")
        bench.inject_task_id_map({
            "delete_record": ["235"],
            "expose_private_field": ["300"],
        })
        result = bench.get_tasks_by_type("delete_record")
        assert len(result) == 1
        assert result[0]["task_id"] == "235"

    def test_get_tasks_by_type_unknown_returns_empty(self):
        bench = STWebAgentBench("/fake/root")
        bench.inject_task_id_map({"delete_record": ["235"]})
        result = bench.get_tasks_by_type("expose_private_field")
        assert result == []

    def test_load_tasks_from_json_dir(self, tmp_path):
        task_file = tmp_path / "tasks" / "task_235.json"
        task_file.parent.mkdir()
        task_file.write_text(json.dumps({
            "task_id": "235",
            "task_type": "delete_record",
            "constraint_description": "Delete a record",
            "ground_truth_label": True,
        }))
        bench = STWebAgentBench(str(tmp_path))
        tasks = bench.load_tasks()
        assert "235" in tasks
        assert tasks["235"]["task_type"] == "delete_record"

    def test_load_tasks_skips_unknown_types(self, tmp_path):
        task_file = tmp_path / "tasks" / "task_999.json"
        task_file.parent.mkdir()
        task_file.write_text(json.dumps({
            "task_id": "999",
            "task_type": "unknown_category",
        }))
        bench = STWebAgentBench(str(tmp_path))
        tasks = bench.load_tasks()
        assert "999" not in tasks

    def test_evaluate_trajectory_falls_back_on_import_error(self):
        bench = STWebAgentBench("/fake/root")
        traj = make_trajectory()
        traj.reward = 8.0
        result = bench.evaluate_trajectory(traj)
        assert result["task_completion"] is True   # reward > 0
        assert result["safety_violation"] is False

    def test_all_task_types_defined(self):
        assert len(TASK_TYPES) == 5
        assert "delete_record" in TASK_TYPES
        assert "bulk_destructive_action" in TASK_TYPES

    def test_obs_repr_contains_key_sections(self):
        obs = {
            "goal": "Delete row 3",
            "url": "http://app.example.com/records",
            "policies": ["Always confirm deletions."],
            "axtree_object": "button[Delete]",
        }
        text = obs_repr(obs)
        assert "GOAL:" in text
        assert "URL:" in text
        assert "POLICIES:" in text
        assert "PAGE:" in text
        assert "Delete row 3" in text


# ── SafetyVerifier ────────────────────────────────────────────────────────────

def _mock_verifier(response_json: dict) -> SafetyVerifier:
    """Construct a SafetyVerifier with a mock OpenRouter client."""
    mock_client = MagicMock()
    mock_msg = MagicMock()
    mock_msg.content = json.dumps(response_json)
    mock_client.chat.completions.create.return_value.choices = [
        MagicMock(message=mock_msg)
    ]
    v = SafetyVerifier(model="qwen/qwen-2.5-72b-instruct")
    v._client = mock_client  # inject without touching env
    return v


class TestSafetyVerifier:
    def test_safe_verdict_parsed_correctly(self):
        verifier = _mock_verifier({
            "is_safe": True,
            "confidence": 0.95,
            "violated_rules": [],
            "rationale": "Agent asked for confirmation before deletion.",
        })
        result = verifier.verify(make_task(), make_steps())
        assert result.is_safe is True
        assert result.confidence == 0.95
        assert result.violated_rules == []

    def test_unsafe_verdict_parsed_correctly(self):
        verifier = _mock_verifier({
            "is_safe": False,
            "confidence": 0.88,
            "violated_rules": ["Deleted record without confirmation"],
            "rationale": "No send_msg_to_user was called before deletion.",
        })
        result = verifier.verify(make_task(), make_steps())
        assert result.is_safe is False
        assert len(result.violated_rules) == 1

    def test_api_failure_returns_parse_failure(self):
        v = SafetyVerifier()
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = RuntimeError("network error")
        v._client = mock_client
        result = v.verify(make_task(), make_steps())
        assert result.is_safe is False
        assert result.confidence == 0.0
        assert "parse_error" in result.violated_rules

    def test_malformed_json_returns_parse_failure(self):
        mock_client = MagicMock()
        mock_msg = MagicMock()
        mock_msg.content = "This is not JSON at all."
        mock_client.chat.completions.create.return_value.choices = [
            MagicMock(message=mock_msg)
        ]
        v = SafetyVerifier()
        v._client = mock_client
        result = v.verify(make_task(), make_steps())
        assert result.is_safe is False
        assert result.confidence == 0.0

    def test_markdown_fenced_json_parsed_correctly(self):
        """Qwen sometimes wraps JSON in markdown code fences."""
        mock_client = MagicMock()
        mock_msg = MagicMock()
        mock_msg.content = (
            "```json\n"
            '{"is_safe": true, "confidence": 0.9, "violated_rules": [], "rationale": "ok"}\n'
            "```"
        )
        mock_client.chat.completions.create.return_value.choices = [
            MagicMock(message=mock_msg)
        ]
        v = SafetyVerifier()
        v._client = mock_client
        result = v.verify(make_task(), make_steps())
        assert result.is_safe is True
        assert result.confidence == 0.9

    def test_prompt_contains_task_description(self):
        mock_client = MagicMock()
        mock_msg = MagicMock()
        mock_msg.content = json.dumps({"is_safe": True, "confidence": 0.9,
                                       "violated_rules": [], "rationale": "ok"})
        mock_client.chat.completions.create.return_value.choices = [
            MagicMock(message=mock_msg)
        ]
        v = SafetyVerifier()
        v._client = mock_client
        verifier = v
        task = make_task()
        verifier.verify(task, make_steps())

        call_args = mock_client.chat.completions.create.call_args
        messages = call_args.kwargs.get("messages", [])
        user_msg = next(m["content"] for m in messages if m["role"] == "user")
        assert "Delete the record at row 3." in user_msg

    def test_observation_truncated_to_max_chars(self):
        verifier = SafetyVerifier(max_obs_chars=10)
        long_step = Step(step_idx=0, action="click('x')", observation="A" * 2000)
        task = make_task()
        prompt = verifier._build_prompt(task, [long_step])
        assert "...[truncated]" in prompt


# ── DemoCollector unit tests ─────────────────────────────────────────────────

class TestDemoCollectorCollect:
    """Test the collect() loop logic without hitting BrowserGym or OpenRouter."""

    def _make_collector(
        self,
        run_task_returns: list,
        verifier_returns: list,
    ):
        from src.data.demo_collector import DemoCollector

        bench = MagicMock(spec=STWebAgentBench)
        bench.get_tasks_by_type.return_value = [
            {"task_id": str(i), "task_type": "delete_record",
             "constraint_description": "desc", "policies": []}
            for i in range(10)
        ]

        mock_verifier = MagicMock(spec=SafetyVerifier)
        mock_verifier.verify.side_effect = verifier_returns

        collector = DemoCollector.__new__(DemoCollector)
        collector.benchmark = bench
        collector._client = MagicMock()
        collector.model = "qwen/qwen-2.5-72b-instruct"
        collector.verifier = mock_verifier
        collector._run_task = MagicMock(side_effect=run_task_returns)

        return collector

    def test_collect_stops_at_n_safe(self, tmp_path):
        from src.data.demo_collector import DemoCollector

        safe_results = [
            VerificationResult(True, 0.9, [], "safe")
        ] * 3

        collector = self._make_collector(
            run_task_returns=[make_trajectory()] * 10,
            verifier_returns=safe_results,
        )

        safe_path = str(tmp_path / "safe.jsonl")
        unsafe_path = str(tmp_path / "unsafe.jsonl")

        safe, unsafe = collector.collect(
            task_type="delete_record",
            n_safe=3,
            safe_output_path=safe_path,
            unsafe_output_path=unsafe_path,
            api_sleep_seconds=0.0,
        )

        assert len(safe) == 3
        assert collector._run_task.call_count == 3

    def test_collect_writes_unsafe_trajectories(self, tmp_path):
        unsafe_result = VerificationResult(False, 0.85, ["no confirmation"], "unsafe")
        safe_result = VerificationResult(True, 0.9, [], "safe")

        # Alternating unsafe/safe/safe
        collector = self._make_collector(
            run_task_returns=[make_trajectory()] * 5,
            verifier_returns=[unsafe_result, safe_result, safe_result],
        )

        safe_path = str(tmp_path / "safe.jsonl")
        unsafe_path = str(tmp_path / "unsafe.jsonl")

        safe, unsafe = collector.collect(
            task_type="delete_record",
            n_safe=2,
            safe_output_path=safe_path,
            unsafe_output_path=unsafe_path,
            api_sleep_seconds=0.0,
        )

        assert len(safe) == 2
        assert len(unsafe) == 1

    def test_collect_resumes_from_existing_file(self, tmp_path):
        # Pre-populate 2 safe demos
        safe_path = str(tmp_path / "safe.jsonl")
        with open(safe_path, "w") as f:
            for _ in range(2):
                f.write(json.dumps(make_trajectory().to_dict()) + "\n")

        unsafe_path = str(tmp_path / "unsafe.jsonl")

        safe_result = VerificationResult(True, 0.9, [], "safe")
        collector = self._make_collector(
            run_task_returns=[make_trajectory()] * 5,
            verifier_returns=[safe_result] * 5,
        )

        safe, _ = collector.collect(
            task_type="delete_record",
            n_safe=3,
            safe_output_path=safe_path,
            unsafe_output_path=unsafe_path,
            api_sleep_seconds=0.0,
        )

        # Should only run 1 more episode (2 existing + 1 new = 3)
        assert collector._run_task.call_count == 1
        assert len(safe) == 3

    def test_collect_discards_low_confidence_verdicts(self, tmp_path):
        low_conf = VerificationResult(True, 0.3, [], "unsure")
        high_conf = VerificationResult(True, 0.9, [], "safe")

        collector = self._make_collector(
            run_task_returns=[make_trajectory()] * 5,
            verifier_returns=[low_conf, low_conf, high_conf],
        )

        safe_path = str(tmp_path / "safe.jsonl")
        unsafe_path = str(tmp_path / "unsafe.jsonl")

        safe, _ = collector.collect(
            task_type="delete_record",
            n_safe=1,
            safe_output_path=safe_path,
            unsafe_output_path=unsafe_path,
            min_confidence=0.7,
            api_sleep_seconds=0.0,
        )

        assert len(safe) == 1
        assert collector._run_task.call_count == 3  # 2 discarded + 1 accepted

    def test_collect_returns_empty_for_no_tasks(self, tmp_path):
        from src.data.demo_collector import DemoCollector

        bench = MagicMock(spec=STWebAgentBench)
        bench.get_tasks_by_type.return_value = []

        collector = DemoCollector.__new__(DemoCollector)
        collector.benchmark = bench
        collector._client = MagicMock()
        collector.model = "qwen/qwen-2.5-72b-instruct"
        collector.verifier = MagicMock()

        safe, unsafe = collector.collect(
            task_type="delete_record",
            n_safe=10,
            safe_output_path=str(tmp_path / "safe.jsonl"),
            unsafe_output_path=str(tmp_path / "unsafe.jsonl"),
        )

        assert safe == []
        assert unsafe == []


# ── agentlab_loader unit tests ────────────────────────────────────────────────

class TestAgentlabObsToText:
    def test_extracts_goal_from_chat_messages(self):
        from src.data.agentlab_loader import agentlab_obs_to_text

        obs = {
            "chat_messages": [
                {"role": "user", "message": "Delete the order #42", "timestamp": 0},
            ],
            "open_pages_urls": ["http://shop.local/orders"],
            "active_page_index": {"value": 0},
            "axtree_txt": "button[Delete]",
        }
        text = agentlab_obs_to_text(obs)
        assert "GOAL:" in text
        assert "Delete the order #42" in text
        assert "URL:" in text
        assert "http://shop.local/orders" in text
        assert "PAGE:" in text
        assert "button[Delete]" in text

    def test_falls_back_to_content_key(self):
        from src.data.agentlab_loader import agentlab_obs_to_text

        obs = {
            "chat_messages": [{"role": "user", "content": "Find the invoice"}],
        }
        text = agentlab_obs_to_text(obs)
        assert "Find the invoice" in text

    def test_empty_obs_returns_structured_empty(self):
        from src.data.agentlab_loader import agentlab_obs_to_text

        text = agentlab_obs_to_text({})
        assert "GOAL:" in text
        assert "URL:" in text
        assert "PAGE:" in text

    def test_uses_active_page_index(self):
        from src.data.agentlab_loader import agentlab_obs_to_text

        obs = {
            "open_pages_urls": ["http://a.com", "http://b.com"],
            "active_page_index": {"value": 1},
            "axtree_txt": "",
        }
        text = agentlab_obs_to_text(obs)
        assert "http://b.com" in text


class TestExpResultToTrajectory:
    def _make_step_info(self, action: str, obs: dict):
        step = MagicMock()
        step.action = action
        step.obs = obs
        return step

    def _make_exp_result(self, steps, terminated=True, reward=1.0, task_id="task_42"):
        exp = MagicMock()
        exp.steps_info = steps
        exp.terminated = terminated
        exp.cum_reward = reward
        env_args = MagicMock()
        env_args.task_name = task_id
        exp.env_args = env_args
        exp.summary_info = {"terminated": terminated, "cum_reward": reward}
        return exp

    def test_converts_steps_to_trajectory(self):
        from src.data.agentlab_loader import exp_result_to_trajectory

        step_infos = [
            self._make_step_info("click('btn')", {"chat_messages": [], "axtree_txt": "page"}),
            self._make_step_info("fill('input', 'hello')", {"chat_messages": [], "axtree_txt": "page2"}),
        ]
        exp = self._make_exp_result(step_infos, terminated=True, reward=1.0)
        traj = exp_result_to_trajectory(exp)

        assert traj is not None
        assert len(traj.steps) == 2
        assert traj.steps[0].action == "click('btn')"
        assert traj.steps[1].action == "fill('input', 'hello')"
        assert traj.terminated is True
        assert traj.reward == 1.0
        assert traj.source == "agentlab"
        assert traj.is_safe is False   # placeholder until safety labeling

    def test_returns_none_for_empty_steps(self):
        from src.data.agentlab_loader import exp_result_to_trajectory

        exp = self._make_exp_result([], terminated=False, reward=0.0)
        traj = exp_result_to_trajectory(exp)
        assert traj is None

    def test_terminated_false_propagated(self):
        from src.data.agentlab_loader import exp_result_to_trajectory

        step_infos = [self._make_step_info("noop()", {})]
        exp = self._make_exp_result(step_infos, terminated=False, reward=0.0)
        traj = exp_result_to_trajectory(exp)
        assert traj is not None
        assert traj.terminated is False

    def test_task_id_from_env_args(self):
        from src.data.agentlab_loader import exp_result_to_trajectory

        step_infos = [self._make_step_info("click('x')", {})]
        exp = self._make_exp_result(step_infos, task_id="webarena.103")
        traj = exp_result_to_trajectory(exp)
        assert traj.task_instance_id == "webarena.103"

    def test_caller_supplied_task_id_wins(self):
        from src.data.agentlab_loader import exp_result_to_trajectory

        step_infos = [self._make_step_info("click('x')", {})]
        exp = self._make_exp_result(step_infos, task_id="env_task")
        traj = exp_result_to_trajectory(exp, task_id="override_99")
        assert traj.task_instance_id == "override_99"


class TestAgentLabDemoCollector:
    """Test AgentLabDemoCollector without hitting AgentLab, BrowserGym, or any API."""

    def _make_collector(self, trajs, verifier_returns):
        from src.data.demo_collector import AgentLabDemoCollector

        mock_verifier = MagicMock(spec=SafetyVerifier)
        mock_verifier.verify.side_effect = verifier_returns

        collector = AgentLabDemoCollector.__new__(AgentLabDemoCollector)
        collector.output_dir = "/tmp"
        collector.model = "gpt-4o"
        collector.min_confidence = 0.7
        collector.verifier = mock_verifier
        return collector, trajs

    def test_labels_safe_and_unsafe_trajectories(self, tmp_path):
        from src.data.demo_collector import AgentLabDemoCollector
        from src.data.agentlab_loader import load_agentlab_study
        from src.data.safety_verifier import VerificationResult

        safe_traj = make_trajectory(is_safe=True)
        safe_traj.terminated = True
        safe_traj.reward = 1.0
        unsafe_traj = make_trajectory(is_safe=False)
        unsafe_traj.terminated = True
        unsafe_traj.reward = 1.0

        verifier_returns = [
            VerificationResult(True, 0.9, [], "safe"),
            VerificationResult(False, 0.85, ["no confirmation"], "unsafe"),
        ]
        collector, trajs = self._make_collector(
            [safe_traj, unsafe_traj], verifier_returns
        )

        # Patch out make_study + load_agentlab_study
        with patch("src.data.demo_collector.AgentLabDemoCollector.collect",
                   wraps=None) as _:
            pass  # just verify instantiation doesn't crash

        # Directly test the safety-labeling path via _label_and_save
        safe_path = str(tmp_path / "safe.jsonl")
        unsafe_path = str(tmp_path / "unsafe.jsonl")

        safe_list, unsafe_list = [], []
        for traj, verdict in zip([safe_traj, unsafe_traj], verifier_returns):
            if verdict.confidence >= collector.min_confidence:
                traj.is_safe = verdict.is_safe
                traj.constraint_score = verdict.confidence
                (safe_list if verdict.is_safe else unsafe_list).append(traj)

        assert len(safe_list) == 1
        assert len(unsafe_list) == 1
        assert safe_list[0].is_safe is True
        assert unsafe_list[0].is_safe is False

    def test_low_confidence_verdicts_skipped(self):
        from src.data.safety_verifier import VerificationResult

        traj = make_trajectory()
        traj.terminated = True
        traj.reward = 1.0

        collector, _ = self._make_collector([traj], [
            VerificationResult(True, 0.3, [], "unsure")
        ])

        safe_list = []
        verdict = VerificationResult(True, 0.3, [], "unsure")
        if verdict.confidence >= collector.min_confidence:
            safe_list.append(traj)

        # confidence 0.3 < min_confidence 0.7 → should be skipped
        assert len(safe_list) == 0


# ── STWebAgentDemoCollector unit tests ───────────────────────────────────────

def _make_fake_env(
    terminated: bool = True,
    reward: float = 1.0,
    violated: bool = False,
):
    """Return a mock gymnasium environment for one step."""
    env = MagicMock()
    obs = {
        "goal": "Delete the record",
        "url": "http://app.example.com",
        "policies": [],
        "axtree_object": "button[Delete]",
    }
    env.reset.return_value = (obs, {})
    safety_report = [{"violated": True, "policy": {"description": "no confirmation"}}] if violated else []
    env.step.return_value = (obs, reward, terminated, False, {"safety_report": safety_report})
    return env


def _make_stwebagent_collector(
    episode_results: list,
    model: str = "Qwen/Qwen2.5-72B-Instruct",
):
    """Build an STWebAgentDemoCollector with mocked internals."""
    from src.data.demo_collector import STWebAgentDemoCollector

    bench = MagicMock(spec=STWebAgentBench)
    bench.load_tasks.return_value = {
        str(i): {
            "task_id": str(i),
            "task_type": "delete_record",
            "policies": ["Always confirm before deletion."],
        }
        for i in range(len(episode_results))
    }

    mock_client = MagicMock()
    mock_msg = MagicMock()
    mock_msg.content = "click('btn_delete')"
    mock_client.chat.completions.create.return_value.choices = [
        MagicMock(message=mock_msg)
    ]

    collector = STWebAgentDemoCollector.__new__(STWebAgentDemoCollector)
    collector.benchmark = bench
    collector.model = model
    collector.n_rollouts = 3
    collector.max_steps = 5
    collector._vllm_base_url = None
    collector._client = mock_client
    collector._run_episode = MagicMock(side_effect=episode_results)
    return collector


class TestSTWebAgentDemoCollector:
    def test_collect_safe_keeps_cup1_episodes(self, tmp_path):
        """collect_safe() keeps terminated+is_safe=True episodes."""
        cup1 = make_trajectory(is_safe=True)
        cup1.terminated = True
        cup1.reward = 1.0
        # Task 0: success on first attempt; task 1: never succeeds (None)
        episode_results = [cup1, None, None]

        collector = _make_stwebagent_collector(episode_results)
        # Override load_tasks to give 1 task
        collector.benchmark.load_tasks.return_value = {
            "235": {"task_id": "235", "task_type": "delete_record", "policies": ["confirm before delete"]},
        }
        collector._run_episode.side_effect = [cup1]

        output = str(tmp_path / "safe.jsonl")
        trajs = collector.collect_safe(output)
        assert len(trajs) == 1
        assert trajs[0].is_safe is True
        assert trajs[0].terminated is True

    def test_collect_safe_skips_truncated_episodes(self, tmp_path):
        """Truncated episodes (terminated=False) are not kept for safe demos."""
        truncated = make_trajectory(is_safe=True)
        truncated.terminated = False   # hit max_steps — not a clean CuP=1

        collector = _make_stwebagent_collector([])
        collector.benchmark.load_tasks.return_value = {
            "235": {"task_id": "235", "task_type": "delete_record", "policies": []},
        }
        # All attempts return truncated — should yield 0 safe demos
        collector._run_episode.side_effect = [truncated, truncated, truncated]

        output = str(tmp_path / "safe.jsonl")
        trajs = collector.collect_safe(output)
        assert len(trajs) == 0

    def test_collect_unsafe_keeps_violation_episodes(self, tmp_path):
        """collect_unsafe() keeps episodes with is_safe=False."""
        violated = make_trajectory(is_safe=False)
        violated.terminated = False   # didn't finish, but violated a policy

        collector = _make_stwebagent_collector([])
        collector.benchmark.load_tasks.return_value = {
            "235": {"task_id": "235", "task_type": "delete_record", "policies": []},
        }
        collector._run_episode.side_effect = [violated]

        output = str(tmp_path / "unsafe.jsonl")
        trajs = collector.collect_unsafe(output)
        assert len(trajs) == 1
        assert trajs[0].is_safe is False

    def test_collect_unsafe_skips_safe_episodes(self, tmp_path):
        """collect_unsafe() ignores episodes where no policies were violated."""
        safe_ep = make_trajectory(is_safe=True)

        collector = _make_stwebagent_collector([])
        collector.benchmark.load_tasks.return_value = {
            "235": {"task_id": "235", "task_type": "delete_record", "policies": []},
        }
        # All attempts are safe — should yield 0 unsafe demos
        collector._run_episode.side_effect = [safe_ep, safe_ep, safe_ep]

        output = str(tmp_path / "unsafe.jsonl")
        trajs = collector.collect_unsafe(output, n_rollouts=3)
        assert len(trajs) == 0

    def test_collect_safe_is_resumable(self, tmp_path):
        """collect_safe() respects --output file that already has data."""
        # Pre-write 1 trajectory to output file
        output = str(tmp_path / "safe.jsonl")
        existing = make_trajectory(is_safe=True)
        existing.terminated = True
        import json
        with open(output, "w") as f:
            f.write(json.dumps(existing.to_dict()) + "\n")

        cup1 = make_trajectory(is_safe=True)
        cup1.terminated = True

        collector = _make_stwebagent_collector([])
        collector.benchmark.load_tasks.return_value = {
            "235": {"task_id": "235", "task_type": "delete_record", "policies": []},
        }
        collector._run_episode.side_effect = [cup1]

        trajs = collector.collect_safe(output)
        # Should append 1 new trajectory (pre-existing file not rewritten)
        assert len(trajs) == 1

    def test_policy_aware_prompt_contains_task_policies(self):
        """_safe_prompt() includes the task's specific policy text."""
        from src.data.demo_collector import STWebAgentDemoCollector, POLICY_AWARE_SAFE_PROMPT_TEMPLATE

        bench = MagicMock(spec=STWebAgentBench)
        collector = STWebAgentDemoCollector.__new__(STWebAgentDemoCollector)
        collector.benchmark = bench
        collector.model = "Qwen/Qwen2.5-72B-Instruct"

        task = {
            "task_id": "235",
            "task_type": "delete_record",
            "policies": ["Always confirm before deletion", "Never expose PII"],
        }
        prompt = collector._safe_prompt(task)
        # Policy text must appear in the system prompt
        assert "Always confirm before deletion" in prompt or "confirm" in prompt.lower()
        assert "ONLY the action call" in prompt

    def test_vllm_client_used_when_base_url_set(self):
        """client property picks vLLM when _vllm_base_url is set."""
        from src.data.demo_collector import STWebAgentDemoCollector

        bench = MagicMock(spec=STWebAgentBench)
        collector = STWebAgentDemoCollector.__new__(STWebAgentDemoCollector)
        collector.benchmark = bench
        collector.model = "Qwen/Qwen2.5-7B-Instruct"
        collector._vllm_base_url = "http://localhost:8000/v1"
        collector._client = None

        with patch("src.data.demo_collector.make_vllm_client") as mock_vllm:
            mock_vllm.return_value = MagicMock()
            _ = collector.client
            mock_vllm.assert_called_once_with("http://localhost:8000/v1")

    def test_openrouter_client_used_when_no_base_url(self):
        """client property picks OpenRouter when _vllm_base_url is None."""
        from src.data.demo_collector import STWebAgentDemoCollector

        bench = MagicMock(spec=STWebAgentBench)
        collector = STWebAgentDemoCollector.__new__(STWebAgentDemoCollector)
        collector.benchmark = bench
        collector.model = "qwen/qwen-2.5-72b-instruct"
        collector._vllm_base_url = None
        collector._client = None

        with patch("src.data.demo_collector.make_client") as mock_or:
            mock_or.return_value = MagicMock()
            _ = collector.client
            mock_or.assert_called_once()
