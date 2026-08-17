# ABOUTME: Defers the benchmark's per-step evaluator run to a single call at episode end.
# ABOUTME: Not run directly — installed by episode_runner when episode.defer_validation is on.
"""
The measured cost, on klogin03 2026-08-15 (scratch/probe_step_breakdown.py):
`_task_validate` is **4.69 s of a 6.6 s step — 71 %**, while every observation
extraction combined is 0.55 s. The benchmark re-runs its task evaluator and all
its policy evaluators after every single action.

Almost all of that is thrown away. `task.validate(page, chat_messages,
trajectory)` is **stateless per call**: it takes the whole trajectory and
recomputes from scratch each time (`ST-WebAgentBench/.../task.py::validate`), so
the verdict after step N is simply superseded by the verdict after step N+1.
Only the last one is ever read — `episode_runner` takes its safety report from
the final step, and the `cup_state` keep rule reads the database for completion
rather than the benchmark's reward.

So this gates `_task_validate` to a cheap no-op during the episode, and the
caller asks for one real evaluation when the episode is over. Same final
verdict, one evaluator run instead of thirty.

What the caller MUST take over, and why this is opt-in rather than default:

  - **Episode termination.** `done` comes out of `validate()` — it is where the
    benchmark notices the agent called `answer()`. Gated, `done` is always
    False, so the loop would run to `max_steps`. `episode_runner` detects the
    answer action itself when this is enabled.
  - **Per-step rewards.** `reward_best` across steps is no longer observable;
    only the final reward is. Nothing in the `cup_state` keep path uses it (the
    database decides completion), but a run that scores from the benchmark's own
    reward should leave this off.

One incidental benefit: `validate()` APPENDS a synthetic "stopped, too many
steps" action to the trajectory whenever `len(trajectory) >= max_steps`. Called
every step, it keeps appending — the trajectory grows on each call once the cap
is reached. Calling it once cannot do that.
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

_GATE_FLAG = "_icrl_validation_deferred"
_REAL_VALIDATE = "_icrl_real_task_validate"

#: What a gated `_task_validate` returns. The shape is load-bearing: `post_step`
#: unpacks four values and then reads `safety_penalty` and `safety_report` off
#: the fourth (custom_env.py:549-568), so a bare `{}` would raise a KeyError.
_QUIET_RESULT = (0.0, False, "", {"safety_penalty": 0.0, "safety_report": []})


def install_deferred_validation(env) -> bool:
    """
    Silence the env's per-step evaluator run until `validate_now` is called.

    Returns True when the gate is in place, False (with a warning) when the env
    exposes no `_task_validate` — a speed optimisation must never be the reason
    a run dies.
    """
    inner = getattr(env, "unwrapped", env)

    if getattr(inner, _GATE_FLAG, False):
        return True

    real = getattr(inner, "_task_validate", None)
    if real is None:
        logger.warning("%s has no _task_validate — per-step validation left on",
                       type(inner).__name__)
        return False

    def gated_task_validate():
        if getattr(inner, _GATE_FLAG, False):
            return _QUIET_RESULT
        return real()

    setattr(inner, _REAL_VALIDATE, real)
    inner._task_validate = gated_task_validate
    setattr(inner, _GATE_FLAG, True)
    logger.debug("per-step validation deferred on %s", type(inner).__name__)
    return True


def validate_now(env) -> tuple:
    """
    Run the real evaluation once and return `(reward, done, user_message, info)`,
    with `info["safety_report"]` populated exactly as a normal step would.

    Safe to call on an env that was never gated: it falls through to the env's
    own `_task_validate`. Re-arms the gate afterwards so a caller that keeps
    stepping does not silently pay per-step validation again.
    """
    inner = getattr(env, "unwrapped", env)
    real = getattr(inner, _REAL_VALIDATE, None) or getattr(inner, "_task_validate", None)
    if real is None:
        logger.warning("%s has no _task_validate — no final verdict available",
                       type(inner).__name__)
        return _QUIET_RESULT

    was_gated = getattr(inner, _GATE_FLAG, False)
    setattr(inner, _GATE_FLAG, False)
    try:
        return real()
    finally:
        setattr(inner, _GATE_FLAG, was_gated)
