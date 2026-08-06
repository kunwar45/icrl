# ABOUTME: Data-collection pipeline: roll out LLM agents in a real benchmark env and
# ABOUTME: keep episodes the ground-truth evaluators label. Run via scripts/collect_trajectories.py.
"""
Config-driven trajectory collection.

One engine, many runs: a YAML in configs/trajectory_collection/ fully defines a
collection run — benchmark, tasks, model/backend, the exact prompt, the episode
mode, and the keep rule. Adding a benchmark means writing one
`<benchmark>_adapter.py`; adding a variant (expert vs unsafe, a new model, a
prompt ablation) means writing one config, never a new script.

"Collection" here means real rollouts labeled by the benchmark's own
evaluators — as opposed to synthetic *generation*, which is a separate concern.
"""
from src.trajectory_collection.benchmark_adapter import BenchmarkAdapter, get_adapter
from src.trajectory_collection.collection_runner import run_collection

__all__ = ["BenchmarkAdapter", "get_adapter", "run_collection"]
