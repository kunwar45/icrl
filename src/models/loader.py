"""Model + tokenizer loading, LoRA-aware."""
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
from peft import PeftModel
from omegaconf import DictConfig


def load_model_and_tokenizer(
    model_name: str,
    cfg: DictConfig,
    causal_lm: bool = True,
    lora_path: str = None,
) -> tuple:
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, cache_dir=cfg.paths.model_cache
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_cls = AutoModelForCausalLM if causal_lm else AutoModel
    model = model_cls.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        cache_dir=cfg.paths.model_cache,
    )

    if lora_path is not None:
        model = PeftModel.from_pretrained(model, lora_path)

    return model, tokenizer
