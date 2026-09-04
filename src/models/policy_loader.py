# ABOUTME: Loads the ODCV policy family (Qwen3.6, a conditional-generation architecture) the way its LoRA adapters expect, and scopes new LoRA targets to the language model
# ABOUTME: Import: from src.models.policy_loader import load_policy_model, lora_target_regex
"""
Qwen3.6-27B's checkpoint is `Qwen3_5ForConditionalGeneration`: the text stack
lives at `model.language_model.*` beside a vision tower. Loading it with
AutoModelForCausalLM renames the modules to `model.layers.*`, and every LoRA
trained on the real layout (the organism's adapter targets
`model\\.language_model\\..*\\.(q_proj|...)$`) then fails to attach. Job 5229152
died exactly there. Load with AutoModelForImageTextToText, as the LASR trainer
does, and give new adapters the same language-model-scoped regex.
"""

from __future__ import annotations

import re


def load_policy_model(name_or_path: str, **kwargs):
    from transformers import AutoModelForCausalLM

    try:
        from transformers import AutoModelForImageTextToText

        return AutoModelForImageTextToText.from_pretrained(name_or_path, **kwargs)
    except Exception as e:  # a plain text-only checkpoint
        print(
            f"AutoModelForImageTextToText refused {name_or_path} ({str(e)[:120]}); using AutoModelForCausalLM"
        )
        return AutoModelForCausalLM.from_pretrained(name_or_path, **kwargs)


def lora_target_regex(model, names: list[str]) -> str | list[str]:
    """Restrict LoRA to the language model when the checkpoint carries a vision tower."""
    if any(".language_model." in n for n, _ in model.named_modules()):
        return (
            r"model\.language_model\..*\.("
            + "|".join(re.escape(n) for n in names)
            + r")$"
        )
    return list(names)
