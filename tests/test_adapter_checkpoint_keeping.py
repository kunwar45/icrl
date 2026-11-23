# ABOUTME: Tests for keeping intermediate adapters: copying only adapter files out of a checkpoint, and mapping every kept adapter to its path in the Hugging Face repo
# ABOUTME: Run: pytest tests/test_adapter_checkpoint_keeping.py -q
from pathlib import Path

from src.lagrangian_finetuning.adapter_checkpoint_keeping import (
    intermediate_adapters,
    keep_adapter_checkpoint,
)


def _checkpoint(d: Path, *names: str) -> Path:
    d.mkdir(parents=True, exist_ok=True)
    for n in names:
        (d / n).write_text(n)
    return d


def test_keeps_adapter_files_and_never_the_optimizer(tmp_path):
    ckpt = _checkpoint(
        tmp_path / "trainer" / "checkpoint-50",
        "adapter_model.safetensors",
        "adapter_config.json",
        "optimizer.pt",
        "scheduler.pt",
        "trainer_state.json",
    )
    kept = keep_adapter_checkpoint(ckpt, tmp_path / "adapter_checkpoints", 50)
    assert kept == tmp_path / "adapter_checkpoints" / "step-00050"
    assert sorted(p.name for p in kept.iterdir()) == [
        "adapter_config.json",
        "adapter_model.safetensors",
    ]
    graft = _checkpoint(
        tmp_path / "trainer" / "checkpoint-100", "graft_adapters.pt", "optimizer.pt"
    )
    assert [
        p.name
        for p in keep_adapter_checkpoint(
            graft, tmp_path / "adapter_checkpoints", 100
        ).iterdir()
    ] == ["graft_adapters.pt"]
    assert (
        keep_adapter_checkpoint(_checkpoint(tmp_path / "empty"), tmp_path / "k", 1)
        is None
    )


def test_repo_paths_for_single_and_multi_round_runs(tmp_path):
    run = tmp_path / "run"
    for step in (50, 100):
        _checkpoint(
            run / "rl" / "adapter_checkpoints" / f"step-{step:05d}",
            "adapter_model.safetensors",
        )
    _checkpoint(
        run / "icrl_round_00" / "rl" / "adapter_checkpoints" / "step-00125",
        "adapter_model.safetensors",
    )
    _checkpoint(
        run / "icrl_round_00" / "rl" / "adapter", "adapter_model.safetensors"
    )  # has copies: not repeated
    _checkpoint(
        run / "icrl_round_01" / "rl" / "adapter", "adapter_model.safetensors"
    )  # older run: no copies
    _checkpoint(run / "icrl_round_02" / "rl" / "adapter")  # no adapter files: skipped
    got = {p for _, p in intermediate_adapters(run)}
    assert got == {
        "checkpoints/step-00050",
        "checkpoints/step-00100",
        "checkpoints/icrl_round_00/step-00125",
        "checkpoints/icrl_round_01/end_of_round",
    }
