# ABOUTME: Runs episode work across worker threads — parallel across chains, sequential within one.
# ABOUTME: Not run directly — used by collection_runner and generation_runner when concurrency > 1.
"""
Why concurrency pays here: an episode step spends most of its wall-clock in the
browser (CDP round-trips, page settling) and only a fraction of it in the model.
Run one episode at a time and the vLLM server sits at batch size 1, so a 4-GPU
72B server idles through every browser step. Several episodes in flight keep
the same server batching, at nearly the same per-request latency — throughput
close to linear in the number of concurrent episodes, on the GPUs already paid
for.

The unit of scheduling is a CHAIN: a list of work items that must run in order.
Chains run concurrently, items inside one chain never overlap. That is what
lets the generation side parallelise safely — tasks whose ground-truth checks
read the same tables go in one chain (so their before/after comparisons stay
attributable), tasks reading different tables go in different chains. Callers
with no such constraint pass one item per chain and get a flat parallel map.

A hand-rolled pool rather than ThreadPoolExecutor because each worker thread
owns two things that MUST be released when it finishes: a Playwright driver
subprocess (see playwright_thread_isolation) and a pooled database connection.
ThreadPoolExecutor offers an `initializer` but no matching finalizer, and a
leaked driver per worker is how a long cluster run runs the node out of
processes.
"""
from __future__ import annotations

import logging
import queue
import threading
from typing import Any, Callable, Iterable, Optional, Sequence, TypeVar

logger = logging.getLogger(__name__)

Item = TypeVar("Item")
Result = TypeVar("Result")


def release_episode_thread_resources() -> None:
    """
    Release everything a worker thread holds open. Safe to call on a thread that
    holds nothing, and never raises — it runs in teardown paths where an
    exception would mask the failure that got us there.
    """
    try:
        from src.environments.playwright_thread_isolation import stop_thread_playwright
        stop_thread_playwright()
    except Exception as e:  # pragma: no cover - teardown must not raise
        logger.debug("releasing this thread's Playwright driver failed: %s", e)
    try:
        from src.trajectory_collection.stwebagentbench_state_verifier import \
            close_thread_connection
        close_thread_connection()
    except Exception as e:  # pragma: no cover - teardown must not raise
        logger.debug("closing this thread's database connection failed: %s", e)


def group_into_chains(items: Iterable[Item],
                      group_of: Optional[Callable[[Item], str]] = None
                      ) -> list[list[Item]]:
    """
    Bucket items into chains by group key, preserving order within each chain.

    `group_of=None` means "nothing collides": every item becomes its own chain
    and the whole set can run at once.
    """
    if group_of is None:
        return [[item] for item in items]
    chains: dict[str, list[Item]] = {}
    for item in items:
        chains.setdefault(group_of(item), []).append(item)
    return list(chains.values())


def run_chains(chains: Sequence[Sequence[Item]],
               run_item: Callable[[Item], Result],
               concurrency: int) -> list[list[Any]]:
    """
    Run `chains` concurrently; items within a chain run in order on one thread.

    Returns results shaped like `chains`. An item that raises contributes the
    exception object in its slot rather than aborting the run — one browser
    dying must not discard the episodes that succeeded alongside it, and the
    callers here already treat a failed episode as "kept nothing".

    `concurrency` is clamped to the number of chains: more workers than chains
    would only start idle Playwright drivers.
    """
    if not chains:
        return []

    workers = max(1, min(int(concurrency), len(chains)))
    results: list[list[Any]] = [[None] * len(chain) for chain in chains]

    if workers == 1:
        # Deliberately not "a pool of one": staying on the calling thread keeps
        # single-threaded runs byte-identical to how they behaved before this
        # module existed, and keeps tracebacks in the caller's stack.
        for chain_index, chain in enumerate(chains):
            for item_index, item in enumerate(chain):
                try:
                    results[chain_index][item_index] = run_item(item)
                except Exception as e:
                    logger.warning("work item %r failed: %s", item, e)
                    results[chain_index][item_index] = e
        return results

    pending: queue.Queue = queue.Queue()
    for chain_index in range(len(chains)):
        pending.put(chain_index)

    def worker() -> None:
        try:
            while True:
                try:
                    chain_index = pending.get_nowait()
                except queue.Empty:
                    return
                chain = chains[chain_index]
                for item_index, item in enumerate(chain):
                    try:
                        results[chain_index][item_index] = run_item(item)
                    except Exception as e:
                        logger.warning("work item %r failed: %s", item, e)
                        results[chain_index][item_index] = e
        finally:
            # Every exit path — drained queue, or an escape no `except` above
            # caught — gives the driver and the connection back.
            release_episode_thread_resources()

    logger.info("running %d chains (%d items) across %d worker threads",
                len(chains), sum(len(c) for c in chains), workers)
    threads = [threading.Thread(target=worker, name=f"episode-worker-{i}",
                                daemon=True)
               for i in range(workers)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    return results
