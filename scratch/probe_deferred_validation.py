# ABOUTME: Login-node probe — does deferring per-step validation change the verdict, and how much faster is it?
# ABOUTME: Run on killarney login node: PYTHONPATH=. python probe_deferred_validation.py
"""
Runs the SAME fixed action sequence twice against a real SuiteCRM episode:
once with the benchmark validating every step (its normal behaviour), once with
validation deferred to a single call at the end. Then compares the final
verdicts.

The speedup is only worth having if the verdict is identical — the safety report
is the ground truth the whole expert set is filtered on.
"""
import json
import os
import time

from src.trajectory_collection.benchmark_adapter import get_adapter

TASK = int(os.environ.get("PROBE_TASK", "236"))


def run(defer: bool) -> dict:
    from src.environments.browsergym_deferred_validation import (
        install_deferred_validation, validate_now)

    adapter = get_adapter({"name": "stwebagentbench", "task_ids": [TASK],
                           "lean_observation": True})
    env = adapter.make_env(TASK, max_steps=10, end_on_score=False, slow_mo_ms=150)
    gated = install_deferred_validation(env) if defer else False
    try:
        adapter.reset(env)
        base = os.environ["WA_SUITECRM"].rstrip("/")
        actions = [f"goto('{base}/#/leads/index')", "noop()", "noop()", "noop()"]

        step_times, info = [], {}
        for action in actions:
            t0 = time.time()
            _obs, reward, _term, _trunc, info = adapter.step(env, action)
            step_times.append(time.time() - t0)

        if gated:
            reward, _done, _msg, info = validate_now(env)

        report = adapter.safety_report(info)
        return {
            "defer": defer,
            "gated": gated,
            "reward": float(reward),
            "step_times": [round(s, 2) for s in step_times],
            "mean_step": round(sum(step_times) / len(step_times), 2),
            "n_policies": len(report),
            "verdicts": sorted(
                (r.get("policy_id", "?"), bool(r.get("violated"))) for r in report),
        }
    finally:
        try:
            env.close()
        except Exception:
            pass
        from src.trajectory_collection.episode_concurrency import \
            release_episode_thread_resources
        release_episode_thread_resources()


if __name__ == "__main__":
    print("running WITH per-step validation (benchmark default)...")
    baseline = run(defer=False)
    print("running WITH validation deferred...")
    deferred = run(defer=True)

    for label, r in (("per-step", baseline), ("deferred", deferred)):
        print(f"\n{label}: mean step {r['mean_step']}s  steps {r['step_times']}")
        print(f"   reward={r['reward']}  policies={r['n_policies']}  gated={r['gated']}")

    same_verdicts = baseline["verdicts"] == deferred["verdicts"]
    same_reward = baseline["reward"] == deferred["reward"]
    speedup = baseline["mean_step"] / deferred["mean_step"] if deferred["mean_step"] else 0

    print(f"\nverdicts identical : {same_verdicts}")
    if not same_verdicts:
        print(f"   per-step: {json.dumps(baseline['verdicts'])}")
        print(f"   deferred: {json.dumps(deferred['verdicts'])}")
    print(f"reward identical   : {same_reward} "
          f"({baseline['reward']} vs {deferred['reward']})")
    print(f"step-time speedup  : {speedup:.2f}x "
          f"({baseline['mean_step']}s -> {deferred['mean_step']}s)")

    ok = same_verdicts and same_reward and deferred["gated"]
    print(f"\nRESULT: {'SAFE TO ENABLE' if ok else 'DO NOT ENABLE — verdict changed'}")
    raise SystemExit(0 if ok else 1)
