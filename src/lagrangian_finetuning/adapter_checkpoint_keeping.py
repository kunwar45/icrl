# ABOUTME: Keeps a copy of the adapter from every trainer checkpoint (adapters only, never optimizer state) under <run>/rl/adapter_checkpoints/step-NNNNN, and lists those copies for publishing
# ABOUTME: Use: trainer.add_callback(KeepAdapterCheckpoints(out_dir / "adapter_checkpoints")); intermediate_adapters(run_dir) gives (folder, path in the HF repo) pairs
"""
The GRPO trainers keep only their newest checkpoint (save_total_limit=1): a full checkpoint carries the
optimizer state (about 2.5 GB for a rank-128 LoRA on a 7B model), so several would fill scratch. The adapter
alone is what is needed to evaluate or publish an intermediate model, so this callback copies just the
adapter files out of each checkpoint as it is written, and the publishing script uploads those copies.
How often a copy is made is the trainer's save_steps.
"""

from __future__ import annotations

import shutil
from pathlib import Path

# LoRA adapters (PEFT) and GRAFT's retain/forget adapters
ADAPTER_FILES = (
    "adapter_model.safetensors",
    "adapter_config.json",
    "graft_adapters.pt",
)


def keep_adapter_checkpoint(
    checkpoint_dir: Path, keep_root: Path, step: int
) -> Path | None:
    """Copy the adapter files of one checkpoint to keep_root/step-NNNNN; None if the checkpoint has none."""
    checkpoint_dir = Path(checkpoint_dir)
    present = [n for n in ADAPTER_FILES if (checkpoint_dir / n).exists()]
    if not present:
        return None
    dst = Path(keep_root) / f"step-{int(step):05d}"
    dst.mkdir(parents=True, exist_ok=True)
    for name in present:
        shutil.copy2(checkpoint_dir / name, dst / name)
    return dst


def keep_adapter_checkpoints_callback(keep_root: Path):
    """A TrainerCallback that runs keep_adapter_checkpoint on the main process after every save."""
    from transformers import TrainerCallback

    class KeepAdapterCheckpoints(TrainerCallback):
        def on_save(self, args, state, control, **kwargs):
            if state.is_world_process_zero:
                ckpt = Path(args.output_dir) / f"checkpoint-{state.global_step}"
                kept = keep_adapter_checkpoint(ckpt, keep_root, state.global_step)
                if kept is not None:
                    print(
                        f"kept adapter of step {state.global_step} in {kept}",
                        flush=True,
                    )
            return control

    return KeepAdapterCheckpoints()


def intermediate_adapters(run_dir: Path) -> list[tuple[Path, str]]:
    """(folder, path in the HF repo) for every intermediate adapter a finished run kept.

    Kept copies: <...>/adapter_checkpoints/step-NNNNN  ->  checkpoints/<prefix>/step-NNNNN, where <prefix>
    is the round directory of a multi-round run (icrl_round_03) or nothing. Multi-round runs from before the
    callback existed have only each round's final adapter: icrl_round_NN/rl/adapter -> checkpoints/
    icrl_round_NN/end_of_round (skipped when that round kept copies).
    """
    run_dir = Path(run_dir)
    out: list[tuple[Path, str]] = []
    rounds_with_copies = set()
    for d in sorted(run_dir.glob("**/adapter_checkpoints/step-*")):
        if not d.is_dir() or not any((d / n).exists() for n in ADAPTER_FILES):
            continue
        rel = d.relative_to(run_dir).parts
        prefix = [p for p in rel[: rel.index("adapter_checkpoints")] if p != "rl"]
        if prefix:
            rounds_with_copies.add(prefix[0])
        out.append((d, "/".join(["checkpoints", *prefix, d.name])))
    for rd in sorted(run_dir.glob("*_round_*")):
        adapter = rd / "rl" / "adapter"
        if rd.name in rounds_with_copies or not adapter.is_dir():
            continue
        if any((adapter / n).exists() for n in ADAPTER_FILES):
            out.append((adapter, f"checkpoints/{rd.name}/end_of_round"))
    return out
