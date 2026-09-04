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


def load_policy_model(name_or_path: str, quantize_4bit: bool = False, **kwargs):
    from transformers import AutoModelForCausalLM

    if quantize_4bit:
        # QLoRA: a 27B in nf4 is ~15 GB, so a 4096-token DPO pair's backward fits on
        # four L40S (bf16 weights + activations did not: job 5230855 OOM'd in backward).
        import torch
        from transformers import BitsAndBytesConfig

        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

    # device_map="auto" packs the last GPU with layers AND the lm_head; a 6k-token
    # DPO/SFT batch then materialises 6144 x 151k logits in fp32 on that GPU beside
    # its layers (job 5229277: "unspecified launch failure" at the third eval pair).
    # Cap every GPU well under its size so the head has room.
    if kwargs.get("device_map") == "auto" and "max_memory" not in kwargs:
        import torch

        n = torch.cuda.device_count()
        if n > 1:
            # 40% of each GPU: a 27B bf16 model then spreads evenly (~13.5 GB per L40S)
            # and every GPU keeps ~26 GB for activations and the fp32 logits TRL builds
            # (job 5229323 OOM'd asking for 10.9 GB on a GPU holding 34 GB of weights).
            total = min(torch.cuda.mem_get_info(i)[1] for i in range(n)) // (1024**3)
            cap = max(8, int(total * 0.40))
            kwargs["max_memory"] = {i: f"{cap}GiB" for i in range(n)}
            kwargs["max_memory"]["cpu"] = "64GiB"

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
