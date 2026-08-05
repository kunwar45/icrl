"""Unit tests for STWebAgentEnv — BrowserGym is mocked."""
from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Fixtures & helpers
# ---------------------------------------------------------------------------

def _make_obs(goal="Delete record", url="https://app.example.com", policies=None,
              axtree="[button 42] Delete\n[button 43] Cancel", chat_messages=None):
    return {
        "goal": goal,
        "url": url,
        "policies": policies or ["Confirm before deletion"],
        "axtree_object": axtree,
        "chat_messages": chat_messages or [],
        "pruned_html": "<html/>",
        "screenshot": b"\x89PNG...",  # should be stripped
    }


def _make_mock_gym_env(obs, reset_info=None, step_reward=0.0, terminated=False, truncated=False):
    mock_env = MagicMock()
    mock_env.reset.return_value = (obs, reset_info or {})
    mock_env.step.return_value = (obs, step_reward, terminated, truncated, {})
    return mock_env


# ---------------------------------------------------------------------------
# Screenshot stripping
# ---------------------------------------------------------------------------

def test_screenshot_stripped_on_reset():
    raw_obs = _make_obs()
    mock_gym_env = _make_mock_gym_env(raw_obs)

    with patch.dict(sys.modules, {"browsergym": MagicMock()}):
        from icrl.envs.stwebagent import STWebAgentEnv, STWebAgentEnvConfig
        with patch("icrl.envs.stwebagent.gym") as mock_gym:
            mock_gym.make.return_value = mock_gym_env
            env = STWebAgentEnv(STWebAgentEnvConfig(task_id="235"))
            obs, _ = env.reset()

    assert "screenshot" not in obs


def test_screenshot_stripped_on_step():
    raw_obs = _make_obs()
    mock_gym_env = _make_mock_gym_env(raw_obs, step_reward=0.0)

    with patch.dict(sys.modules, {"browsergym": MagicMock()}):
        from icrl.envs.stwebagent import STWebAgentEnv, STWebAgentEnvConfig
        with patch("icrl.envs.stwebagent.gym") as mock_gym:
            mock_gym.make.return_value = mock_gym_env
            env = STWebAgentEnv(STWebAgentEnvConfig(task_id="235"))
            env._gym_env = mock_gym_env
            env._step_count = 0
            obs, _, _, _, _ = env.step("click('42')")

    assert "screenshot" not in obs


# ---------------------------------------------------------------------------
# Shaped reward arithmetic
# ---------------------------------------------------------------------------

def test_step_cost_applied():
    raw_obs = _make_obs()
    mock_gym_env = _make_mock_gym_env(raw_obs, step_reward=0.0, terminated=False)

    with patch.dict(sys.modules, {"browsergym": MagicMock()}):
        from icrl.envs.stwebagent import STWebAgentEnv, STWebAgentEnvConfig
        with patch("icrl.envs.stwebagent.gym") as mock_gym:
            mock_gym.make.return_value = mock_gym_env
            cfg = STWebAgentEnvConfig(task_id="235", step_cost=0.1, completion_bonus=10.0)
            env = STWebAgentEnv(cfg)
            env._gym_env = mock_gym_env
            env._step_count = 0
            _, reward, _, _, _ = env.step("click('42')")

    assert reward == pytest.approx(-0.1)


def test_completion_bonus_on_success():
    raw_obs = _make_obs()
    mock_gym_env = _make_mock_gym_env(raw_obs, step_reward=1.0, terminated=True)

    with patch.dict(sys.modules, {"browsergym": MagicMock()}):
        from icrl.envs.stwebagent import STWebAgentEnv, STWebAgentEnvConfig
        with patch("icrl.envs.stwebagent.gym") as mock_gym:
            mock_gym.make.return_value = mock_gym_env
            cfg = STWebAgentEnvConfig(task_id="235", step_cost=0.1, completion_bonus=10.0)
            env = STWebAgentEnv(cfg)
            env._gym_env = mock_gym_env
            env._step_count = 0
            _, reward, terminated, _, _ = env.step("finish('done')")

    assert reward == pytest.approx(10.0 - 0.1)
    assert terminated is True


def test_no_bonus_on_failure():
    raw_obs = _make_obs()
    mock_gym_env = _make_mock_gym_env(raw_obs, step_reward=0.0, terminated=True)

    with patch.dict(sys.modules, {"browsergym": MagicMock()}):
        from icrl.envs.stwebagent import STWebAgentEnv, STWebAgentEnvConfig
        with patch("icrl.envs.stwebagent.gym") as mock_gym:
            mock_gym.make.return_value = mock_gym_env
            cfg = STWebAgentEnvConfig(task_id="235", step_cost=0.1, completion_bonus=10.0)
            env = STWebAgentEnv(cfg)
            env._gym_env = mock_gym_env
            env._step_count = 0
            _, reward, _, _, _ = env.step("finish('failed')")

    assert reward == pytest.approx(-0.1)


# ---------------------------------------------------------------------------
# obs_repr content
# ---------------------------------------------------------------------------

def test_obs_repr_contains_goal_url_policies_page():
    with patch.dict(sys.modules, {"browsergym": MagicMock()}):
        from icrl.envs.stwebagent import STWebAgentEnv, STWebAgentEnvConfig
        with patch("icrl.envs.stwebagent.gym"):
            env = STWebAgentEnv(STWebAgentEnvConfig(task_id="235"))

    obs = _make_obs(
        goal="My goal",
        url="https://x.com",
        policies=["Policy A"],
        axtree="[button 1] Click",
    )
    text = env.obs_repr(obs)
    assert "My goal" in text
    assert "https://x.com" in text
    assert "Policy A" in text
    assert "[button 1] Click" in text


def test_axtree_truncated_at_2000():
    with patch.dict(sys.modules, {"browsergym": MagicMock()}):
        from icrl.envs.stwebagent import STWebAgentEnv, STWebAgentEnvConfig
        with patch("icrl.envs.stwebagent.gym"):
            env = STWebAgentEnv(STWebAgentEnvConfig(task_id="235"))

    obs = _make_obs(axtree="x" * 5000)
    text = env.obs_repr(obs)
    assert "truncated" in text
    assert len(text) < 10000  # not blowing up


# ---------------------------------------------------------------------------
# obs_dim / n_actions sentinels
# ---------------------------------------------------------------------------

def test_obs_dim_n_actions_return_sentinel():
    with patch.dict(sys.modules, {"browsergym": MagicMock()}):
        from icrl.envs.stwebagent import STWebAgentEnv, STWebAgentEnvConfig
        with patch("icrl.envs.stwebagent.gym"):
            env = STWebAgentEnv(STWebAgentEnvConfig(task_id="235"))
    assert env.obs_dim == -1
    assert env.n_actions == -1
