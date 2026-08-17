# ABOUTME: Unit tests for the contrastive-dataset driver's counting, deficit and saturation logic
# ABOUTME: Run: pytest tests/test_build_contrastive_dataset.py -q (no cluster, no SLURM)
"""
The driver's job control is exercised without SLURM: Side.counts_by_task reads
the filesystem, and everything that decides "run another pass or stop" is pure.
"""
import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location(
    "build_contrastive_dataset", REPO_ROOT / "scripts/build_contrastive_dataset.py")
driver = importlib.util.module_from_spec(spec)
sys.modules["build_contrastive_dataset"] = driver
spec.loader.exec_module(driver)


def _side(tmp_path: Path, target: int, traces: dict[int, int]):
    """A Side whose output dir contains `traces` = {task_id: how many}."""
    out = tmp_path / "set"
    out.mkdir(parents=True, exist_ok=True)
    for task_id, n in traces.items():
        for i in range(n):
            (out / f"task_{task_id}_trace_{i}.json").write_text("{}")
    side = driver.Side.__new__(driver.Side)          # bypass config loading
    side.name, side.target, side.output_dir = "expert", target, out
    side.saturated, side.empty_passes, side.passes_run = False, 0, 0
    return side


def test_counts_traces_per_task(tmp_path):
    side = _side(tmp_path, 150, {236: 3, 237: 2})
    assert side.counts_by_task() == {236: 3, 237: 2}
    assert side.total() == 5


def test_ignores_non_trace_files(tmp_path):
    side = _side(tmp_path, 150, {236: 1})
    (side.output_dir / "manifest.json").write_text("{}")
    (side.output_dir / "summary.csv").write_text("x")
    (side.output_dir / "summary_pass_123.csv").write_text("x")
    assert side.total() == 1, "only task_*_trace_*.json counts"


def test_deficit_floors_at_zero(tmp_path):
    assert _side(tmp_path, 10, {236: 4}).deficit() == 6
    assert _side(tmp_path, 3, {236: 4}).deficit() == 0


def test_missing_output_dir_counts_as_empty(tmp_path):
    side = _side(tmp_path, 150, {})
    side.output_dir = tmp_path / "not_created_yet"
    assert side.total() == 0 and side.deficit() == 150


def test_report_flags_unmatched_tasks(tmp_path, capsys):
    """Unmatched tasks are the failure that makes a good AUROC meaningless, so the
    report has to call them out rather than just printing totals."""
    expert = _side(tmp_path / "e", 10, {236: 2, 237: 2})
    unsafe = _side(tmp_path / "u", 10, {236: 2, 999: 2})
    unsafe.name = "unsafe"
    driver.report([expert, unsafe], final=True)
    out = capsys.readouterr().out
    assert "matched pairs" in out
    assert "236" in out
    assert "WARNING" in out and "task identity" in out


def test_report_grades_by_size(tmp_path, capsys):
    driver.report([_side(tmp_path / "a", 150, {236: 3})], final=False)
    assert "batch" in capsys.readouterr().out          # below MIN_RUNNABLE
    driver.report([_side(tmp_path / "b", 150, {236: 60})], final=False)
    assert "AUROC still noisy" in capsys.readouterr().out
    driver.report([_side(tmp_path / "c", 150, {236: 130})], final=False)
    assert "trustworthy" in capsys.readouterr().out


def test_job_finished_needs_sacct_confirmation(monkeypatch):
    """An empty squeue result alone must not be read as 'finished' — a dropped
    connection looks identical."""
    calls = []

    class _Result:
        def __init__(self, stdout): self.stdout = stdout

    def fake_run(cmd, **kwargs):
        calls.append(cmd[0])
        if cmd[0] == "squeue":
            return _Result("")            # not queued
        return _Result("")                # sacct also silent -> unknown

    monkeypatch.setattr(driver.subprocess, "run", fake_run)
    assert driver.job_finished("1") is False
    assert "sacct" in calls, "must confirm against sacct"

    def fake_run_done(cmd, **kwargs):
        return _Result("" if cmd[0] == "squeue" else "COMPLETED")
    monkeypatch.setattr(driver.subprocess, "run", fake_run_done)
    assert driver.job_finished("1") is True

    def fake_run_running(cmd, **kwargs):
        return _Result("RUNNING" if cmd[0] == "squeue" else "RUNNING")
    monkeypatch.setattr(driver.subprocess, "run", fake_run_running)
    assert driver.job_finished("1") is False
