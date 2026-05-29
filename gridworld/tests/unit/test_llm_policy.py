"""Unit tests for LLMPolicy — Together AI client is mocked."""
from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import pytest


def _mock_together_response(content: str):
    msg = MagicMock()
    msg.content = content
    choice = MagicMock()
    choice.message = msg
    resp = MagicMock()
    resp.choices = [choice]
    return resp


def _make_obs(goal="Complete task", url="https://example.com", chat_messages=None):
    return {
        "goal": goal,
        "url": url,
        "policies": ["Confirm before deletion"],
        "axtree_object": "[button 1] Submit",
        "chat_messages": chat_messages or [],
        "pruned_html": "<html/>",
    }


# ---------------------------------------------------------------------------
# Basic action generation
# ---------------------------------------------------------------------------

def test_act_returns_string():
    from icrl.policies.llm_policy import LLMPolicy, LLMPolicyConfig

    policy = LLMPolicy(LLMPolicyConfig())
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = _mock_together_response("click('btn-1')")
    policy._client = mock_client

    obs = _make_obs()
    action = policy.act(obs)
    assert action == "click('btn-1')"


def test_act_strips_whitespace():
    from icrl.policies.llm_policy import LLMPolicy, LLMPolicyConfig

    policy = LLMPolicy(LLMPolicyConfig())
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = _mock_together_response("  fill('inp', 'hello')  \n")
    policy._client = mock_client

    action = policy.act(_make_obs())
    assert action == "fill('inp', 'hello')"


# ---------------------------------------------------------------------------
# Prompt content checks
# ---------------------------------------------------------------------------

def test_prompt_contains_goal_and_policies():
    from icrl.policies.llm_policy import LLMPolicy, LLMPolicyConfig

    policy = LLMPolicy(LLMPolicyConfig())
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = _mock_together_response("click('x')")
    policy._client = mock_client

    obs = _make_obs(goal="Delete all records")
    policy.act(obs)

    call_args = mock_client.chat.completions.create.call_args
    messages = call_args.kwargs["messages"]
    user_content = next(m["content"] for m in messages if m["role"] == "user")
    assert "Delete all records" in user_content
    assert "Confirm before deletion" in user_content


def test_system_prompt_mentions_send_msg_to_user():
    from icrl.policies.llm_policy import SYSTEM_PROMPT
    assert "send_msg_to_user" in SYSTEM_PROMPT
    assert "confirmation" in SYSTEM_PROMPT.lower()


# ---------------------------------------------------------------------------
# Constraint warning injection
# ---------------------------------------------------------------------------

def test_constraint_warning_appears_when_cost_exceeds_threshold():
    from icrl.policies.llm_policy import LLMPolicy, LLMPolicyConfig

    mock_constraint = MagicMock()
    import torch
    mock_constraint.cost.return_value = torch.tensor(5.0)

    policy = LLMPolicy(LLMPolicyConfig(constraint_cost_threshold=2.0))
    policy._constraint = mock_constraint
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = _mock_together_response("click('x')")
    policy._client = mock_client

    # Non-empty chat_messages prevents step_history reset at episode boundary
    obs = _make_obs(chat_messages=[{"role": "user", "message": "go"}])
    policy._step_history = [(obs, "click('y')")]  # non-empty history triggers cost check
    policy.act(obs)

    call_args = mock_client.chat.completions.create.call_args
    messages = call_args.kwargs["messages"]
    user_content = next(m["content"] for m in messages if m["role"] == "user")
    assert "CONSTRAINT WARNING" in user_content or "constraint" in user_content.lower()


def test_no_constraint_warning_when_cost_below_threshold():
    from icrl.policies.llm_policy import LLMPolicy, LLMPolicyConfig

    mock_constraint = MagicMock()
    import torch
    mock_constraint.cost.return_value = torch.tensor(0.5)

    policy = LLMPolicy(LLMPolicyConfig(constraint_cost_threshold=2.0))
    policy._constraint = mock_constraint
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = _mock_together_response("click('x')")
    policy._client = mock_client

    obs = _make_obs()
    policy._step_history = [(obs, "click('y')")]
    policy.act(obs)

    call_args = mock_client.chat.completions.create.call_args
    messages = call_args.kwargs["messages"]
    user_content = next(m["content"] for m in messages if m["role"] == "user")
    assert "CONSTRAINT WARNING" not in user_content


# ---------------------------------------------------------------------------
# Episode boundary: step_history reset when chat_messages is empty
# ---------------------------------------------------------------------------

def test_step_history_reset_on_empty_chat():
    from icrl.policies.llm_policy import LLMPolicy, LLMPolicyConfig

    policy = LLMPolicy(LLMPolicyConfig())
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = _mock_together_response("click('x')")
    policy._client = mock_client

    obs_with_history = _make_obs(chat_messages=[{"role": "user", "message": "hi"}])
    policy.act(obs_with_history)
    assert len(policy._step_history) == 1

    obs_new_episode = _make_obs(chat_messages=[])
    policy.act(obs_new_episode)
    assert len(policy._step_history) == 1  # reset then appended once


# ---------------------------------------------------------------------------
# update() is a no-op
# ---------------------------------------------------------------------------

def test_update_returns_empty_dict():
    from icrl.policies.llm_policy import LLMPolicy, LLMPolicyConfig
    policy = LLMPolicy(LLMPolicyConfig())
    result = policy.update([])
    assert result == {}


# ---------------------------------------------------------------------------
# set_constraint resets step history
# ---------------------------------------------------------------------------

def test_set_constraint_resets_step_history():
    from icrl.policies.llm_policy import LLMPolicy, LLMPolicyConfig
    policy = LLMPolicy(LLMPolicyConfig())
    policy._step_history = [("obs", "action"), ("obs2", "action2")]
    policy.set_constraint(None)
    assert policy._step_history == []
