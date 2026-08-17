# ABOUTME: Tests the concurrency machinery — chain grouping, worker pool, thread-local Playwright, lean obs
# ABOUTME: Run: pytest tests/test_episode_concurrency.py -q  (no browser, no database, no network)
"""
What these pin down, in the order the pipeline depends on it:

  1. Chain grouping — items that share a collision group land in ONE chain, in
     order. This is what keeps two tasks reading the same tables from ever
     overlapping, which is the difference between a verified trace and an
     unattributable one.
  2. The worker pool — chains really do overlap, items within a chain never do,
     a raising item does not take the run down with it, and every worker
     releases its thread-local resources on the way out.
  3. The BrowserGym patches — both of which exist to survive the fact that
     ST-WebAgentBench does NOT use BrowserGym's env class. The Playwright
     accessor must be rebound in every module that imported it by value, and
     lean observation must patch extraction functions rather than re-derive an
     observation dict it does not own. Each of those is a bug that reached the
     cluster on 2026-08-15 and is pinned here.
"""
from __future__ import annotations

import threading
import time
import types

import pytest

from src.trajectory_collection.episode_concurrency import (group_into_chains,
                                                           run_chains)


# ── Chain grouping ────────────────────────────────────────────────────────────

def test_no_group_function_means_everything_can_run_at_once():
    chains = group_into_chains([1, 2, 3])
    assert chains == [[1], [2], [3]]


def test_items_sharing_a_group_form_one_ordered_chain():
    # 236/246 both read `leads`, 237/247 both read `opportunities`.
    tasks = [236, 237, 244, 246, 247]
    groups = {236: "leads", 246: "leads",
              237: "opportunities", 247: "opportunities", 244: "cases"}
    chains = group_into_chains(tasks, group_of=lambda t: groups[t])

    assert sorted(map(sorted, chains)) == [[236, 246], [237, 247], [244]]
    for chain in chains:
        assert chain == sorted(chain), "order within a chain must be preserved"


# ── The worker pool ───────────────────────────────────────────────────────────

def test_results_keep_the_shape_and_order_of_the_chains():
    chains = [[1, 2], [3], [4, 5, 6]]
    assert run_chains(chains, lambda x: x * 10, concurrency=3) == \
        [[10, 20], [30], [40, 50, 60]]


def test_single_concurrency_runs_strictly_in_order():
    seen: list[int] = []
    run_chains([[1, 2], [3, 4]], lambda x: seen.append(x), concurrency=1)
    assert seen == [1, 2, 3, 4]


def test_items_in_one_chain_never_overlap():
    """The whole point of chaining: two tasks whose database checks collide must
    not be in flight together, however high concurrency is set."""
    in_flight = 0
    max_in_flight = 0
    lock = threading.Lock()

    def work(_item):
        nonlocal in_flight, max_in_flight
        with lock:
            in_flight += 1
            max_in_flight = max(max_in_flight, in_flight)
        time.sleep(0.02)
        with lock:
            in_flight -= 1

    run_chains([[1, 2, 3, 4]], work, concurrency=8)
    assert max_in_flight == 1


def test_separate_chains_do_overlap():
    """And the converse: independent chains must actually run at the same time,
    or concurrency buys nothing."""
    barrier = threading.Barrier(4, timeout=10)

    def work(_item):
        # Times out and raises if fewer than 4 items are ever in flight at once.
        barrier.wait()
        return "ran"

    results = run_chains([[1], [2], [3], [4]], work, concurrency=4)
    assert results == [["ran"], ["ran"], ["ran"], ["ran"]]


def test_a_failing_item_does_not_lose_the_others():
    def work(item):
        if item == 2:
            raise RuntimeError("browser died")
        return item

    results = run_chains([[1], [2], [3]], work, concurrency=3)
    assert results[0] == [1] and results[2] == [3]
    assert isinstance(results[1][0], RuntimeError)


def test_a_failing_item_still_runs_the_rest_of_its_chain():
    attempted: list[int] = []

    def work(item):
        attempted.append(item)
        if item == 1:
            raise RuntimeError("first revision blew up")
        return item

    results = run_chains([[1, 2]], work, concurrency=1)
    assert attempted == [1, 2]
    assert isinstance(results[0][0], RuntimeError) and results[0][1] == 2


def test_workers_release_their_thread_resources(monkeypatch):
    """A leaked Playwright driver per worker is how a long cluster run runs the
    node out of processes, so teardown is asserted, not assumed."""
    released: list[str] = []
    monkeypatch.setattr(
        "src.trajectory_collection.episode_concurrency.release_episode_thread_resources",
        lambda: released.append(threading.current_thread().name))

    run_chains([[1], [2], [3]], lambda x: x, concurrency=3)
    assert len(released) == 3
    assert len(set(released)) == 3, "each worker must release its own resources"


def test_more_workers_than_chains_starts_no_spare_threads(monkeypatch):
    released: list[str] = []
    monkeypatch.setattr(
        "src.trajectory_collection.episode_concurrency.release_episode_thread_resources",
        lambda: released.append(threading.current_thread().name))

    run_chains([[1], [2]], lambda x: x, concurrency=16)
    assert len(released) == 2


def test_empty_input_is_not_an_error():
    assert run_chains([], lambda x: x, concurrency=4) == []


# ── BrowserGym: thread-local Playwright ───────────────────────────────────────

def _install_fake_browsergym(monkeypatch, extra_modules=()):
    """Fake browsergym whose accessor is held by value in several modules, the
    way `browsergym.core.env` and the benchmark fork's `custom_env` both do."""
    import sys

    from src.environments import playwright_thread_isolation

    # A fresh install per test: the real flag lives on the module being faked.
    playwright_thread_isolation._original_accessor.clear()
    playwright_thread_isolation._local.playwright = None

    def original():
        return "original"

    core = types.ModuleType("browsergym.core")
    core._PLAYWRIGHT = None
    core._get_global_playwright = original
    core._set_global_playwright = lambda pw: None
    monkeypatch.setitem(sys.modules, "browsergym.core", core)

    holders = {}
    for name in ("browsergym.core.env",) + tuple(extra_modules):
        module = types.ModuleType(name)
        module._get_global_playwright = original          # bound BY VALUE
        monkeypatch.setitem(sys.modules, name, module)
        holders[name] = module

    return playwright_thread_isolation, core, holders


def test_thread_isolation_rebinds_every_module_holding_the_accessor(monkeypatch):
    """The bug this pins: several modules do
    `from browsergym.core import _get_global_playwright`, binding the function
    OBJECT at import time. Patching only the package leaves them on the
    original — which reads the module global `_PLAYWRIGHT` that the replaced
    setter no longer writes, so it returns None and every episode dies in
    reset() with 'NoneType' object has no attribute 'selectors'.

    Hand-listing the sites is what missed the benchmark fork's own env module,
    so the patch sweeps sys.modules instead.
    """
    isolation, core, holders = _install_fake_browsergym(
        monkeypatch,
        # The site the hand-written list missed on killarney, 2026-08-15.
        extra_modules=("stwebagentbench.browser_env.custom_env",))

    isolation.install_thread_isolated_playwright()

    assert core._get_global_playwright is isolation._thread_local_playwright
    for name, module in holders.items():
        assert module._get_global_playwright is isolation._thread_local_playwright, \
            f"{name} still holds the original accessor"


def test_a_module_imported_after_installation_is_caught_on_the_next_call(monkeypatch):
    """make_env re-asserts the patch because the fork's env module may only be
    imported when gym.make runs — after the first install."""
    import sys

    isolation, _core, _holders = _install_fake_browsergym(monkeypatch)
    isolation.install_thread_isolated_playwright()

    # A module that grabbed the original before the sweep could see it.
    latecomer = types.ModuleType("stwebagentbench.browser_env.custom_env")
    latecomer._get_global_playwright = isolation._original_accessor[0]
    monkeypatch.setitem(sys.modules, "stwebagentbench.browser_env.custom_env",
                        latecomer)
    assert latecomer._get_global_playwright is not isolation._thread_local_playwright

    isolation.install_thread_isolated_playwright()   # what make_env does
    assert latecomer._get_global_playwright is isolation._thread_local_playwright


def test_unrelated_modules_are_left_alone(monkeypatch):
    import sys

    isolation, _core, _holders = _install_fake_browsergym(monkeypatch)
    bystander = types.ModuleType("somebody_elses_module")
    sentinel = lambda: "not ours"
    bystander._get_global_playwright = sentinel
    monkeypatch.setitem(sys.modules, "somebody_elses_module", bystander)

    isolation.install_thread_isolated_playwright()
    assert bystander._get_global_playwright is sentinel


def test_an_already_running_driver_is_adopted_not_orphaned(monkeypatch):
    isolation, core, _holders = _install_fake_browsergym(monkeypatch)
    sentinel = object()
    core._PLAYWRIGHT = sentinel

    isolation.install_thread_isolated_playwright()

    # The driver BrowserGym already started becomes this thread's own, rather
    # than being left running while a second one is created beside it.
    assert isolation._thread_local_playwright() is sentinel
    isolation._local.playwright = None


# ── BrowserGym: lean observation ──────────────────────────────────────────────
# The first version replaced `_get_obs` with a copy of BrowserGym's, which broke
# on the cluster because ST-WebAgentBench defines its OWN `_get_obs` with a
# different observation dict. These pin the replacement approach: patch the
# extraction functions in whatever namespace the env's `_get_obs` uses, and
# never re-derive somebody else's observation shape.

def _fake_env_module(name="fake_browser_env", with_read_page=False):
    """A module shaped like an env module: extraction functions plus a class
    whose `_get_obs` resolves them from this namespace."""
    import sys

    module = types.ModuleType(name)
    module.extract_screenshot = lambda page: "REAL-SCREENSHOT"
    module.extract_dom_snapshot = lambda page: {"documents": ["real"], "strings": ["real"]}
    module.extract_dom_extra_properties = lambda dom, scale_factor=1.0: {"real": True}
    module.extract_focused_element_bid = lambda page: "bid-42"
    module.extract_merged_axtree = lambda page: {"nodes": ["a", "b"]}
    module._pre_extract = lambda page, tags=None: None
    module._post_extract = lambda page: None

    env_source = """
class FakeEnv:
    use_raw_page_output = False
    def _get_obs(self):
        _pre_extract(self.page, None)
        dom = extract_dom_snapshot(self.page)
        obs = {
            "axtree_object": extract_merged_axtree(self.page),
            "screenshot": extract_screenshot(self.page),
            "dom_object": dom,
            "extra_element_properties": extract_dom_extra_properties(dom, scale_factor=1.0),
            "focused_element_bid": extract_focused_element_bid(self.page),
            "goal": "delete the lead",
            "policies": ["confirm first"],
            "url": "http://crm/#/leads/index",
        }
        _post_extract(self.page)
        if hasattr(self, "read_webpage_content"):
            obs["read_page"] = self.read_webpage_content()
        return obs
"""
    exec(env_source, module.__dict__)
    if with_read_page:
        module.FakeEnv.read_webpage_content = lambda self: "REAL-PAGE-TEXT"
    module.FakeEnv.__module__ = name
    module.FakeEnv._get_obs.__module__ = name
    sys.modules[name] = module
    return module


def test_lean_observation_stubs_the_unused_extractions(monkeypatch):
    from src.environments.browsergym_lean_observation import install_lean_observation

    module = _fake_env_module("fake_env_stub_check")
    monkeypatch.setitem(__import__("sys").modules, module.__name__, module)
    env = module.FakeEnv()
    env.page = object()

    before = env._get_obs()
    assert before["screenshot"] == "REAL-SCREENSHOT"

    assert install_lean_observation(env) is True
    after = env._get_obs()

    # The axtree — the only thing the prompts read — is untouched...
    assert after["axtree_object"] == {"nodes": ["a", "b"]}
    # ...and every key the env builds is still there, with the unused ones empty.
    assert set(after) >= set(before)
    assert after["dom_object"] == {"documents": [], "strings": []}
    assert after["extra_element_properties"] == {}
    assert after["focused_element_bid"] == ""
    assert after["screenshot"].shape == (1, 1, 3)


def test_lean_observation_preserves_fork_specific_keys(monkeypatch):
    """The regression: the fork's obs has `policies` and no `goal_object`.
    Patching extractors rather than rebuilding the dict keeps those intact."""
    from src.environments.browsergym_lean_observation import install_lean_observation

    module = _fake_env_module("fake_env_fork_shape", with_read_page=True)
    monkeypatch.setitem(__import__("sys").modules, module.__name__, module)
    env = module.FakeEnv()
    env.page = object()

    assert install_lean_observation(env) is True
    obs = env._get_obs()

    assert obs["goal"] == "delete the lead"
    assert obs["policies"] == ["confirm first"]
    assert obs["url"] == "http://crm/#/leads/index"
    # the per-step networkidle page read is stubbed, but the key survives
    assert obs["read_page"] == ""


def test_lean_observation_is_idempotent_per_module(monkeypatch):
    from src.environments.browsergym_lean_observation import install_lean_observation

    module = _fake_env_module("fake_env_idempotent")
    monkeypatch.setitem(__import__("sys").modules, module.__name__, module)
    env = module.FakeEnv()
    env.page = object()

    assert install_lean_observation(env) is True
    stub = module.extract_screenshot
    assert install_lean_observation(env) is True
    assert module.extract_screenshot is stub, "re-installing must not re-wrap"


def test_lean_observation_declines_on_a_non_browsergym_env():
    from src.environments.browsergym_lean_observation import install_lean_observation

    class NotAnEnv:
        pass

    # A speed optimisation must never be the reason a run dies.
    assert install_lean_observation(NotAnEnv()) is False


def test_lean_observation_leaves_raw_page_mode_alone(monkeypatch):
    from src.environments.browsergym_lean_observation import install_lean_observation

    module = _fake_env_module("fake_env_raw_page")
    monkeypatch.setitem(__import__("sys").modules, module.__name__, module)
    env = module.FakeEnv()
    env.use_raw_page_output = True

    assert install_lean_observation(env) is False
    assert module.extract_screenshot(None) == "REAL-SCREENSHOT"


def test_lean_observation_unwraps_a_gymnasium_wrapper(monkeypatch):
    from src.environments.browsergym_lean_observation import install_lean_observation

    module = _fake_env_module("fake_env_wrapped")
    monkeypatch.setitem(__import__("sys").modules, module.__name__, module)
    inner = module.FakeEnv()
    inner.page = object()

    class Wrapper:
        unwrapped = inner

    assert install_lean_observation(Wrapper()) is True
    assert module.extract_screenshot(None).shape == (1, 1, 3)


# ── Deferred validation ───────────────────────────────────────────────────────
# The benchmark re-runs every evaluator after every action (measured: 4.69s of a
# 6.6s step) even though validate() is stateless per call and only the last run
# is ever read. These pin the gate and the contract it must honour.

class _FakeValidatingEnv:
    """Shaped like the fork's env: `_task_validate` returns the 4-tuple that
    post_step unpacks, and counts how often it really ran."""
    def __init__(self):
        self.real_calls = 0
        self.trajectory = []

    def _task_validate(self):
        self.real_calls += 1
        return (1.0, True, "done",
                {"safety_penalty": 0.5,
                 "safety_report": [{"policy_id": "consent", "violated": True}]})


def test_gated_validation_does_not_run_the_evaluator():
    from src.environments.browsergym_deferred_validation import \
        install_deferred_validation

    env = _FakeValidatingEnv()
    assert install_deferred_validation(env) is True

    for _ in range(30):
        env._task_validate()
    assert env.real_calls == 0, "the whole point: no evaluator runs mid-episode"


def test_gated_result_has_the_shape_post_step_unpacks():
    """post_step does `reward, done, msg, task_info = self._task_validate()` and
    then reads task_info['safety_penalty'] and ['safety_report'] — a bare {}
    would KeyError inside the benchmark."""
    from src.environments.browsergym_deferred_validation import \
        install_deferred_validation

    env = _FakeValidatingEnv()
    install_deferred_validation(env)
    reward, done, message, info = env._task_validate()

    assert (reward, done, message) == (0.0, False, "")
    assert info["safety_penalty"] == 0.0
    assert info["safety_report"] == []


def test_validate_now_runs_the_real_evaluator_once():
    from src.environments.browsergym_deferred_validation import (
        install_deferred_validation, validate_now)

    env = _FakeValidatingEnv()
    install_deferred_validation(env)
    for _ in range(10):
        env._task_validate()

    reward, done, _message, info = validate_now(env)
    assert env.real_calls == 1
    assert reward == 1.0 and done is True
    assert info["safety_report"] == [{"policy_id": "consent", "violated": True}]


def test_the_gate_is_rearmed_after_a_final_validation():
    from src.environments.browsergym_deferred_validation import (
        install_deferred_validation, validate_now)

    env = _FakeValidatingEnv()
    install_deferred_validation(env)
    validate_now(env)
    env._task_validate()          # a caller that keeps stepping
    assert env.real_calls == 1, "must not silently resume per-step validation"


def test_installing_twice_does_not_stack_gates():
    from src.environments.browsergym_deferred_validation import (
        install_deferred_validation, validate_now)

    env = _FakeValidatingEnv()
    install_deferred_validation(env)
    install_deferred_validation(env)
    validate_now(env)
    assert env.real_calls == 1


def test_validate_now_works_on_an_ungated_env():
    from src.environments.browsergym_deferred_validation import validate_now

    env = _FakeValidatingEnv()
    reward, _done, _msg, _info = validate_now(env)
    assert env.real_calls == 1 and reward == 1.0


def test_gate_declines_on_an_env_without_task_validate():
    from src.environments.browsergym_deferred_validation import \
        install_deferred_validation

    class NotAnEnv:
        pass

    assert install_deferred_validation(NotAnEnv()) is False


def test_gate_unwraps_a_gymnasium_wrapper():
    from src.environments.browsergym_deferred_validation import (
        install_deferred_validation, validate_now)

    inner = _FakeValidatingEnv()

    class Wrapper:
        unwrapped = inner

    wrapper = Wrapper()
    assert install_deferred_validation(wrapper) is True
    inner._task_validate()
    assert inner.real_calls == 0
    validate_now(wrapper)
    assert inner.real_calls == 1


# ── Task metadata cache ───────────────────────────────────────────────────────
# Reading a task's goal + policies boots a browser and logs in: ~40s of a ~150s
# generation cycle, re-reading fixed task configuration, and the flakiest step
# in the pipeline under concurrency. These pin the cache that removes it.

class _CountingAdapter:
    """A real STWebAgentBenchAdapter instance (its __init__ imports browsergym
    and installs patches, so it is bypassed) with the base-class read counted."""
    def __new__(cls, cache_dir, payload=None):
        from src.trajectory_collection.stwebagentbench_adapter import \
            STWebAgentBenchAdapter
        adapter = STWebAgentBenchAdapter.__new__(STWebAgentBenchAdapter)
        adapter.cfg = {"metadata_cache_dir": str(cache_dir)}
        adapter.reads = 0
        adapter._payload = payload or {"goal": "delete the lead",
                                       "policies_block": "confirm first",
                                       "action_space": "click/fill"}
        return adapter


def _patch_super_read(monkeypatch, adapter):
    """Make the base-class read observable and offline."""
    def fake_super_read(self, _task_id):
        self.reads += 1
        return self._payload
    monkeypatch.setattr(
        "src.trajectory_collection.benchmark_adapter.BenchmarkAdapter.task_metadata",
        fake_super_read)


def test_metadata_is_read_once_and_then_cached(tmp_path, monkeypatch):
    adapter = _CountingAdapter(tmp_path / "cache")
    _patch_super_read(monkeypatch, adapter)

    first = adapter.task_metadata(236)
    second = adapter.task_metadata(236)

    assert first == second == adapter._payload
    assert adapter.reads == 1, "the second cycle must not boot a browser again"


def test_a_fresh_process_reuses_the_cache_on_disk(tmp_path, monkeypatch):
    """Each cycle is a NEW python process, so an in-memory cache would buy
    nothing — the win only exists if it survives to disk."""
    first = _CountingAdapter(tmp_path / "cache")
    _patch_super_read(monkeypatch, first)
    first.task_metadata(236)

    second = _CountingAdapter(tmp_path / "cache")
    _patch_super_read(monkeypatch, second)
    second.task_metadata(236)

    assert second.reads == 0


def test_each_task_is_cached_separately(tmp_path, monkeypatch):
    adapter = _CountingAdapter(tmp_path / "cache")
    _patch_super_read(monkeypatch, adapter)

    adapter.task_metadata(236)
    adapter.task_metadata(237)
    adapter.task_metadata(236)

    assert adapter.reads == 2


def test_an_incomplete_cache_entry_is_ignored(tmp_path, monkeypatch):
    """A truncated write would otherwise poison every later cycle silently."""
    import json

    cache_dir = tmp_path / "cache"
    cache_dir.mkdir(parents=True)
    (cache_dir / "task_236.json").write_text(json.dumps({"goal": ""}))

    adapter = _CountingAdapter(cache_dir)
    _patch_super_read(monkeypatch, adapter)
    result = adapter.task_metadata(236)

    assert adapter.reads == 1
    assert result == adapter._payload


def test_corrupt_cache_falls_back_to_reading(tmp_path, monkeypatch):
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir(parents=True)
    (cache_dir / "task_236.json").write_text("{not json")

    adapter = _CountingAdapter(cache_dir)
    _patch_super_read(monkeypatch, adapter)
    assert adapter.task_metadata(236) == adapter._payload
    assert adapter.reads == 1


def test_an_empty_read_is_not_cached(tmp_path, monkeypatch):
    """Caching a failed read would make the failure permanent."""
    adapter = _CountingAdapter(tmp_path / "cache",
                               payload={"goal": "", "policies_block": "",
                                        "action_space": ""})
    _patch_super_read(monkeypatch, adapter)

    adapter.task_metadata(236)
    adapter.task_metadata(236)

    assert adapter.reads == 2
    assert not (tmp_path / "cache" / "task_236.json").exists()
