# ABOUTME: Login-node probe — times each phase inside one env.step to find what actually costs 6.6s
# ABOUTME: Run on killarney login node: PYTHONPATH=. python probe_step_breakdown.py
"""
The lean-observation change bought only 7%, so the per-step cost is somewhere
else. This wraps every phase the fork's step() goes through and prints a
breakdown, so the next optimisation targets the real bottleneck instead of
another guess.
"""
import os
import time
from collections import defaultdict

TIMINGS = defaultdict(list)


def timed(label, fn):
    def wrapper(*args, **kwargs):
        t0 = time.time()
        try:
            return fn(*args, **kwargs)
        finally:
            TIMINGS[label].append(time.time() - t0)
    return wrapper


def main():
    import stwebagentbench.browser_env.custom_env as fork_env

    # Instrument the extraction + validation phases in the fork's namespace.
    for name in ("_pre_extract", "_post_extract", "extract_merged_axtree",
                 "extract_dom_snapshot", "extract_screenshot",
                 "extract_dom_extra_properties", "extract_focused_element_bid"):
        if hasattr(fork_env, name):
            setattr(fork_env, name, timed(name, getattr(fork_env, name)))

    env_class = fork_env.BrowserEnv
    for method in ("_get_obs", "_task_validate", "_wait_dom_loaded",
                   "read_webpage_content", "_active_page_check",
                   "_wait_for_user_message", "pre_step", "post_step"):
        if hasattr(env_class, method):
            setattr(env_class, method, timed(method, getattr(env_class, method)))

    from src.trajectory_collection.benchmark_adapter import get_adapter

    task = int(os.environ.get("PROBE_TASK", "236"))
    adapter = get_adapter({"name": "stwebagentbench", "task_ids": [task],
                           "lean_observation": True})
    env = adapter.make_env(task, max_steps=8, end_on_score=False, slow_mo_ms=150)
    try:
        t0 = time.time()
        obs = adapter.reset(env)
        print(f"reset: {time.time() - t0:.1f}s")
        TIMINGS.clear()

        base = os.environ["WA_SUITECRM"].rstrip("/")
        actions = [f"goto('{base}/#/leads/index')", "noop()", "noop()"]
        for action in actions:
            t0 = time.time()
            obs, *_ = adapter.step(env, action)
            print(f"step {action[:34]:36s} {time.time() - t0:6.2f}s")

        print("\n--- phase totals over 3 steps (seconds) ---")
        for label, times in sorted(TIMINGS.items(), key=lambda kv: -sum(kv[1])):
            print(f"  {label:30s} total {sum(times):6.2f}  "
                  f"calls {len(times):3d}  mean {sum(times)/len(times):5.2f}")

        print(f"\npre_observation_delay = "
              f"{getattr(env.unwrapped, 'pre_observation_delay', 'n/a')}")
        print(f"slow_mo               = {getattr(env.unwrapped, 'slow_mo', 'n/a')}")
    finally:
        try:
            env.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
