#!/usr/bin/env python3
"""
Block A: Collect safe demonstrations from reference orchestrator.

Usage:
    python scripts/collect_demos.py +compute=local run_name=collect_v1
"""
import hydra
from omegaconf import DictConfig
from src.data.st_webagent import STWebAgentBench, TASK_TYPES
from src.data.demo_collector import DemoCollector
from src.utils.config import resolve_paths
from src.utils.compute import seed_everything


@hydra.main(config_path="../configs", config_name="base", version_base=None)
def main(cfg: DictConfig):
    seed_everything(cfg.seed)
    resolve_paths(cfg)

    benchmark = STWebAgentBench(cfg.paths.benchmark_root)
    collector = DemoCollector(benchmark)

    for task_type in TASK_TYPES:
        import os
        output_path = os.path.join(cfg.paths.data_root, "demos", f"{task_type}_safe.jsonl")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        collector.collect(
            task_type=task_type,
            n_demos=cfg.constraint.training.n_safe_demos_per_task,
            output_path=output_path,
        )


if __name__ == "__main__":
    main()
