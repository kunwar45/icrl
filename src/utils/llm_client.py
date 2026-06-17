"""
LLM client factories.

Three backends, all exposing the same interface:
    client.chat.completions.create(model=..., messages=[...], max_tokens=..., temperature=...)
    → response.choices[0].message.content  (str)

Backends
--------
  openrouter   Cloud API via OpenRouter.  Needs OPENROUTER_API_KEY.
               Useful for quick tests without local GPU.  Do not use for bulk runs.
               client = make_openrouter_client()

  hf-local     Local HuggingFace model loaded with transformers.pipeline.
               Works on CPU, MPS (Apple Silicon), or CUDA.
               On multi-GPU SLURM nodes it shards with device_map="auto".
               client = make_hf_client("Qwen/Qwen2.5-1.5B-Instruct")

  vllm         OpenAI-compatible REST server started separately (vllm serve ...).
               Use this for 72B on SLURM — best throughput.
               client = make_vllm_client("http://localhost:8000/v1")

Usage
-----
    from src.utils.llm_client import make_hf_client, make_vllm_client

    # Local test (M1 Mac, small model):
    client = make_hf_client("Qwen/Qwen2.5-1.5B-Instruct")

    # SLURM (vLLM server running on same node):
    client = make_vllm_client("http://localhost:8000/v1")

    response = client.chat.completions.create(
        model="Qwen/Qwen2.5-72B-Instruct",   # ignored by hf-local
        messages=[{"role": "user", "content": "Hello"}],
        max_tokens=128,
        temperature=0.0,
    )
    print(response.choices[0].message.content)
"""
from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

# ── Model name constants ──────────────────────────────────────────────────────

# HuggingFace model IDs — use for hf-local and vllm backends
QWEN_72B_HF = "Qwen/Qwen2.5-72B-Instruct"
QWEN_14B_HF = "Qwen/Qwen2.5-14B-Instruct"
QWEN_7B_HF  = "Qwen/Qwen2.5-7B-Instruct"
QWEN_3B_HF  = "Qwen/Qwen2.5-3B-Instruct"
QWEN_15B_HF = "Qwen/Qwen2.5-1.5B-Instruct"   # good local test model on M1

# OpenRouter slugs — only used with openrouter backend (legacy, avoid for new code)
QWEN_72B_OR = "qwen/qwen-2.5-72b-instruct"
QWEN_7B_OR  = "qwen/qwen-2.5-7b-instruct"

# Keep old names for backward compatibility with other scripts that import them
QWEN_72B      = QWEN_72B_OR
QWEN_7B       = QWEN_7B_OR
QWEN_72B_LOCAL = QWEN_72B_HF
QWEN_7B_LOCAL  = QWEN_7B_HF
QWEN_14B_LOCAL = QWEN_14B_HF


# ── HF-local backend ──────────────────────────────────────────────────────────

class _MockMessage:
    def __init__(self, content: str):
        self.content = content


class _MockChoice:
    def __init__(self, content: str):
        self.message = _MockMessage(content)


class _MockResponse:
    def __init__(self, content: str):
        self.choices = [_MockChoice(content)]


class _HFCompletionsAPI:
    def __init__(self, pipe, model_name: str):
        self._pipe = pipe
        self._model_name = model_name

    def create(
        self,
        model: str,
        messages: list[dict],
        max_tokens: int = 512,
        temperature: float = 0.0,
        **kwargs: Any,
    ) -> _MockResponse:
        do_sample = temperature > 0.0
        gen_kwargs: dict = {
            "max_new_tokens": max_tokens,
            "do_sample": do_sample,
            "pad_token_id": self._pipe.tokenizer.eos_token_id,
        }
        if do_sample:
            gen_kwargs["temperature"] = temperature

        output = self._pipe(messages, **gen_kwargs)

        # transformers text-generation pipeline with chat input:
        # output[0]["generated_text"] is the full list of messages including
        # the new assistant turn appended at the end.
        generated = output[0]["generated_text"]
        if isinstance(generated, list):
            content = generated[-1]["content"]
        else:
            content = str(generated)

        return _MockResponse(content)


class _HFChatNamespace:
    def __init__(self, pipe, model_name: str):
        self.completions = _HFCompletionsAPI(pipe, model_name)


class HFLocalClient:
    """
    Local HuggingFace inference client with an OpenAI-compatible interface.

    The model is loaded once at construction and reused for all calls.
    On Apple Silicon (MPS) it runs bfloat16 on-device.
    On SLURM with multiple GPUs it shards automatically.
    """

    def __init__(self, model_name: str, device: str | None = None):
        import torch
        from transformers import pipeline

        # Device selection
        if device is not None:
            use_device_map = False
            target_device: str | None = device
        elif torch.cuda.is_available():
            n = torch.cuda.device_count()
            if n > 1:
                use_device_map = True
                target_device = None
                logger.info("HFLocalClient: %d GPUs detected — using device_map='auto'", n)
            else:
                use_device_map = False
                target_device = "cuda:0"
        elif torch.backends.mps.is_available():
            use_device_map = False
            target_device = "mps"
        else:
            use_device_map = False
            target_device = "cpu"
            logger.warning("HFLocalClient: no GPU found — running on CPU (will be slow)")

        logger.info("Loading %s ...", model_name)
        self._pipe = pipeline(
            "text-generation",
            model=model_name,
            dtype=torch.bfloat16,
            device_map="auto" if use_device_map else None,
            device=target_device if not use_device_map else None,
            trust_remote_code=True,
        )
        logger.info("Model loaded on %s", target_device or "auto-sharded")
        self.chat = _HFChatNamespace(self._pipe, model_name)
        self._model_name = model_name

    def __repr__(self) -> str:
        return f"HFLocalClient(model={self._model_name!r})"


# ── vLLM backend ──────────────────────────────────────────────────────────────

def make_vllm_client(base_url: str):
    """
    Return an OpenAI-SDK client pointed at a running vLLM server.

    The model name in create() must match the model served by vLLM, e.g.:
        vllm serve Qwen/Qwen2.5-72B-Instruct --tensor-parallel-size 4 --port 8000
        client = make_vllm_client("http://localhost:8000/v1")
        client.chat.completions.create(model="Qwen/Qwen2.5-72B-Instruct", ...)
    """
    from openai import OpenAI
    return OpenAI(base_url=base_url, api_key="EMPTY")


# ── OpenRouter backend (legacy) ───────────────────────────────────────────────

def make_openrouter_client():
    """Return an OpenAI-SDK client pointed at OpenRouter. Needs OPENROUTER_API_KEY."""
    from openai import OpenAI
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "OPENROUTER_API_KEY is not set. "
            "Add it to .env or switch to --backend hf-local / vllm."
        )
    return OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)


# Alias kept for scripts that still call make_client()
make_client = make_openrouter_client


# ── Convenience factory ───────────────────────────────────────────────────────

def make_hf_client(
    model_name: str = QWEN_15B_HF,
    device: str | None = None,
) -> HFLocalClient:
    """
    Load a local HuggingFace chat model and return a client.

    Args:
        model_name:  HuggingFace model ID.  Default: Qwen/Qwen2.5-1.5B-Instruct.
                     For the full pipeline: Qwen/Qwen2.5-72B-Instruct (use vllm instead).
        device:      "cuda", "mps", "cpu", or None (auto-detect).
    """
    return HFLocalClient(model_name=model_name, device=device)
