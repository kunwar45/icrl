# ABOUTME: Login-node probe — does a REAL browser episode work with lean observation, and with two at once?
# ABOUTME: Run on killarney login node: PYTHONPATH=. python probe_lean_and_threads.py
"""
Validates the two changes that unit tests cannot reach, without a GPU:

  1. lean observation produces a usable axtree from the real SuiteCRM SPA
     (the DOM snapshot / screenshot / extra-properties extractions are skipped —
     if the axtree depends on any of them, this is where it shows)
  2. two real browsers driven from two threads both work, which is the first
     time this pipeline has ever run more than one Playwright at a time

No LLM is called: the env is driven directly with fixed actions.
"""
import logging
import os
import sys
import threading
import time

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")

from browsergym.utils.obs import flatten_axtree_to_str  # noqa: E402

from src.trajectory_collection.benchmark_adapter import get_adapter  # noqa: E402

TASK = int(os.environ.get("PROBE_TASK", "236"))


def run_one(tag: str, lean: bool, results: dict) -> None:
    started = time.time()
    adapter = get_adapter({"name": "stwebagentbench", "task_ids": [TASK],
                           "lean_observation": lean})
    env = None
    try:
        env = adapter.make_env(TASK, max_steps=5, end_on_score=False, slow_mo_ms=150)
        t_make = time.time()
        obs = adapter.reset(env)
        t_reset = time.time()

        fields = adapter.prompt_fields(obs)
        axtree = fields["axtree"]

        step_times = []
        for action in ("noop()", "noop()"):
            t0 = time.time()
            obs, _r, _term, _trunc, _info = adapter.step(env, action)
            step_times.append(time.time() - t0)
        axtree_after = flatten_axtree_to_str(obs.get("axtree_object", {}))

        results[tag] = {
            "ok": True,
            "lean": lean,
            "url": fields["url"],
            "axtree_chars": len(axtree),
            "axtree_chars_after_step": len(axtree_after),
            "has_bids": "[" in axtree and "]" in axtree,
            "goal_chars": len(fields["goal"]),
            "policies_chars": len(fields["policies_block"]),
            "make_s": round(t_make - started, 1),
            "reset_s": round(t_reset - t_make, 1),
            "step_s": [round(s, 2) for s in step_times],
            "sample": axtree[:300].replace("\n", " | "),
        }
    except Exception as e:
        import traceback
        results[tag] = {"ok": False, "lean": lean, "error": f"{type(e).__name__}: {e}",
                        "traceback": traceback.format_exc()[-1500:]}
    finally:
        if env is not None:
            try:
                env.close()
            except Exception:
                pass
        from src.trajectory_collection.episode_concurrency import \
            release_episode_thread_resources
        release_episode_thread_resources()


def report(title: str, results: dict) -> None:
    print(f"\n===== {title} =====")
    for tag, r in sorted(results.items()):
        if not r["ok"]:
            print(f"  {tag}: FAILED — {r['error']}")
            print(r["traceback"])
            continue
        print(f"  {tag}: OK  lean={r['lean']}  url={r['url']}")
        print(f"      axtree {r['axtree_chars']} chars (after step: "
              f"{r['axtree_chars_after_step']}), bids present: {r['has_bids']}")
        print(f"      goal {r['goal_chars']} chars, policies {r['policies_chars']} chars")
        print(f"      make {r['make_s']}s  reset {r['reset_s']}s  steps {r['step_s']}s")
        print(f"      axtree head: {r['sample']}")


if __name__ == "__main__":
    # ---- 1. lean vs full, sequential, so the axtree can be compared ----
    sequential: dict = {}
    run_one("full", lean=False, results=sequential)
    run_one("lean", lean=True, results=sequential)
    report("sequential: full vs lean observation", sequential)

    if sequential.get("lean", {}).get("ok") and sequential.get("full", {}).get("ok"):
        lean_chars = sequential["lean"]["axtree_chars"]
        full_chars = sequential["full"]["axtree_chars"]
        ratio = lean_chars / full_chars if full_chars else 0
        print(f"\n  axtree size lean/full = {ratio:.3f} "
              f"({lean_chars} vs {full_chars}) — must be ~1.0")
        lean_step = sum(sequential["lean"]["step_s"])
        full_step = sum(sequential["full"]["step_s"])
        print(f"  step time  lean/full = {lean_step / full_step:.2f} "
              f"({lean_step:.2f}s vs {full_step:.2f}s) — the speedup")

    # ---- 2. two real browsers, two threads, at once ----
    concurrent: dict = {}
    threads = [threading.Thread(target=run_one, args=(f"thread{i}", True, concurrent))
               for i in range(2)]
    t0 = time.time()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    report(f"concurrent: 2 threads, {time.time() - t0:.0f}s wall", concurrent)

    failed = [t for t, r in {**sequential, **concurrent}.items() if not r["ok"]]
    print(f"\nRESULT: {'FAILED — ' + ', '.join(failed) if failed else 'all probes passed'}")
    sys.exit(1 if failed else 0)
