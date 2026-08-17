# ABOUTME: Makes BrowserGym's process-global Playwright instance thread-local so episodes can run concurrently.
# ABOUTME: Not run directly — installed by stwebagentbench_adapter before any concurrent episode starts.
"""
BrowserGym keeps ONE Playwright instance for the whole process
(`browsergym.core._PLAYWRIGHT`, handed out by `_get_global_playwright()`).
Playwright's *sync* API is built on a greenlet loop owned by the thread that
called `sync_playwright().start()`, and using its objects from another thread
raises `greenlet.error: cannot switch to a different thread`. So the global is
exactly what stops N episodes from running in a thread pool — one browser per
episode is fine, one Playwright driver shared across threads is not.

The fix Playwright itself prescribes is one Playwright instance per thread.
This module swaps the accessor for a thread-local one, so each worker thread
lazily starts (and later stops) its own driver, and every browser, context and
page a thread creates stays inside that thread.

Two details that are easy to get wrong:

  - **Several modules import the accessor BY VALUE.** `browsergym.core.env` and
    ST-WebAgentBench's own `stwebagentbench/browser_env/custom_env.py` both do
    `from browsergym.core import _get_global_playwright`, which binds the
    function OBJECT at import time. Patching only
    `browsergym.core._get_global_playwright` leaves those modules calling the
    original.

    That is not merely "the patch does nothing there". The original reads the
    module global `_PLAYWRIGHT`, which it populates through
    `_set_global_playwright` — also replaced here. So an unpatched caller starts
    a driver, hands it to a setter that writes thread-local state, then returns
    the still-`None` global, and the episode dies with
    `AttributeError: 'NoneType' object has no attribute 'selectors'` inside
    `reset()`. Observed on killarney 2026-08-15, on the full and lean paths
    alike, before any episode ran.

    Hand-listing the import sites is what produced that bug, so this module does
    not: it walks `sys.modules` and rebinds every module holding the original.
    Modules imported AFTER installation pick up the patched function naturally.

  - A driver process per thread leaks if nobody stops it. `stop_thread_playwright()`
    is what a worker calls on its way out; `episode_concurrency` does this in
    its `finally` block, so a crashed episode still releases its driver.
"""
from __future__ import annotations

import logging
import threading

logger = logging.getLogger(__name__)

_INSTALLED_FLAG = "_icrl_thread_isolated_playwright"

_local = threading.local()
#: Every driver this process started, so a leak is observable in a log line
#: rather than only as a slow climb in process count.
_started_by_thread: dict[int, object] = {}
_registry_lock = threading.Lock()


def _thread_local_playwright():
    """Drop-in `browsergym.core._get_global_playwright` — one driver per thread."""
    playwright = getattr(_local, "playwright", None)
    if playwright is None:
        import playwright.sync_api
        playwright = playwright.sync_api.sync_playwright().start()
        _local.playwright = playwright
        with _registry_lock:
            _started_by_thread[threading.get_ident()] = playwright
        logger.debug("started a Playwright driver for thread %s",
                     threading.current_thread().name)
    return playwright


def _thread_local_set_playwright(playwright) -> None:
    """Drop-in `browsergym.core._set_global_playwright`, kept so any caller of
    the upstream setter writes into the calling thread rather than clobbering
    another thread's driver."""
    _local.playwright = playwright
    with _registry_lock:
        _started_by_thread[threading.get_ident()] = playwright


#: The accessor as BrowserGym defined it, captured before the first patch so
#: modules that bound it by value can be recognised by identity.
_original_accessor: list = []


def _rebind_import_sites() -> list[str]:
    """
    Point every module holding the ORIGINAL accessor at the thread-local one.

    Returns the module names rebound, so a missed site is a log line rather than
    an `AttributeError` fifteen minutes into an episode.
    """
    import sys

    if not _original_accessor:
        return []
    original = _original_accessor[0]

    rebound = []
    for module_name, module in list(sys.modules.items()):
        if module is None:
            continue
        try:
            held = getattr(module, "_get_global_playwright", None)
        except Exception:  # pragma: no cover - modules with exotic __getattr__
            continue
        if held is original:
            setattr(module, "_get_global_playwright", _thread_local_playwright)
            rebound.append(module_name)
    return rebound


def install_thread_isolated_playwright() -> None:
    """
    Make BrowserGym hand out a per-thread Playwright instance.

    Idempotent, and safe to call from the main thread after BrowserGym has
    already started its global driver: that instance is adopted as the calling
    thread's own, so the patch never orphans a running driver or starts a
    redundant second one.

    Re-running the `sys.modules` sweep on every call is deliberate and cheap —
    it costs a dictionary walk and closes the window where a module imported
    between two calls still holds the original.
    """
    import browsergym.core as browsergym_core

    if getattr(browsergym_core, _INSTALLED_FLAG, False):
        _rebind_import_sites()
        return

    _original_accessor.append(browsergym_core._get_global_playwright)

    existing = getattr(browsergym_core, "_PLAYWRIGHT", None)
    if existing is not None and getattr(_local, "playwright", None) is None:
        _thread_local_set_playwright(existing)

    browsergym_core._get_global_playwright = _thread_local_playwright
    browsergym_core._set_global_playwright = _thread_local_set_playwright

    rebound = _rebind_import_sites()
    setattr(browsergym_core, _INSTALLED_FLAG, True)
    logger.info("BrowserGym Playwright is now thread-local (rebound in: %s) — "
                "episodes may run concurrently",
                ", ".join(rebound) if rebound else "browsergym.core only")


def stop_thread_playwright() -> None:
    """
    Stop and forget this thread's Playwright driver, if it started one.

    Called by whatever owns the thread's lifetime. Never raises: this runs in
    cleanup paths where an exception would mask the real failure.
    """
    playwright = getattr(_local, "playwright", None)
    if playwright is None:
        return
    _local.playwright = None
    with _registry_lock:
        _started_by_thread.pop(threading.get_ident(), None)
    try:
        playwright.stop()
    except Exception as e:  # pragma: no cover - driver teardown races
        logger.debug("stopping this thread's Playwright driver failed: %s", e)


def live_driver_count() -> int:
    """How many Playwright drivers this process currently holds — for logging
    and for tests that assert cleanup actually happened."""
    with _registry_lock:
        return len(_started_by_thread)
