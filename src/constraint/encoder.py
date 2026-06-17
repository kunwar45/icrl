"""
Trajectory encoder: frozen base model + small MLP head → C_θ(τ) ∈ [0, 1].

Design: mean-pool the last hidden state of the frozen base model (the same
model being fine-tuned). Only the MLP head is ever updated.

Higher score = safer trajectory.

Factory
-------
Use load_qwen_encoder() to get a ready-to-use encoder with a frozen Qwen
backbone. On SLURM this maps the model across all visible GPUs automatically.
"""
from __future__ import annotations

import logging
import os

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase

logger = logging.getLogger(__name__)

# Default small Qwen model: ~3 GB in bfloat16, fits on one A100 / L40S with
# plenty of room for batches.  Base (not Instruct) gives better general
# representations for mean-pool embeddings.
DEFAULT_ENCODER_MODEL = "Qwen/Qwen2.5-1.5B"


class TrajectoryEncoder(nn.Module):
    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        max_length: int = 2048,
        head_hidden: int = 256,
    ):
        super().__init__()
        # Freeze the backbone — only the head will be trained
        for p in model.parameters():
            p.requires_grad_(False)
        self.backbone = model

        self.tokenizer = tokenizer
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.max_length = max_length

        hidden_size = model.config.hidden_size
        self.head = nn.Sequential(
            nn.Linear(hidden_size, head_hidden),
            nn.GELU(),
            nn.Linear(head_hidden, 1),
            nn.Sigmoid(),
        )

    @torch.no_grad()
    def _mean_pool(self, texts: list[str]) -> torch.Tensor:
        """Tokenize, run backbone, mean-pool last hidden state over real tokens."""
        # Pre-truncate text so the tokenizer doesn't process hundreds of thousands
        # of chars for very long (axtree-heavy) trajectories.  ~8 chars/token is
        # a safe upper bound for mixed HTML/natural-language content.
        max_chars = self.max_length * 8
        texts = [t[:max_chars] for t in texts]

        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
            padding=True,
        )
        device = next(self.backbone.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        outputs = self.backbone(**inputs)
        last_hidden = outputs.last_hidden_state          # (B, L, H)

        mask = inputs["attention_mask"].unsqueeze(-1).float()
        pooled = (last_hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        return pooled.float()                            # (B, H)

    @torch.no_grad()
    def embed_texts(self, texts: list[str], batch_size: int = 8) -> torch.Tensor:
        """
        Pre-compute mean-pooled embeddings for a list of texts in mini-batches.

        Since the backbone is frozen, embeddings never change during training.
        Call this once before the training loop and pass the cached tensors to
        constraint_update — avoids re-tokenising and re-running the backbone on
        every gradient step.

        Returns: (N, hidden_size) float tensor on the backbone's device.
        """
        parts = []
        for i in range(0, len(texts), batch_size):
            parts.append(self._mean_pool(texts[i : i + batch_size]))
        return torch.cat(parts, dim=0)

    def forward(self, trajectory_texts: list[str]) -> torch.Tensor:
        """Returns shape (B,) scores in [0, 1]."""
        pooled = self._mean_pool(trajectory_texts)
        return self.head(pooled).squeeze(-1)

    @property
    def hidden_size(self) -> int:
        return self.backbone.config.hidden_size


# ── Factory ───────────────────────────────────────────────────────────────────

def load_qwen_encoder(
    model_name: str = DEFAULT_ENCODER_MODEL,
    max_length: int = 2048,
    head_hidden: int = 256,
    device: str | torch.device | None = None,
    attn_implementation: str | None = None,
) -> TrajectoryEncoder:
    """
    Load a frozen Qwen base model and return a TrajectoryEncoder.

    Args:
        model_name:           HuggingFace model ID.  Default: Qwen/Qwen2.5-1.5B.
        max_length:           Token limit per trajectory (truncated before backbone).
        head_hidden:          Hidden units in the 2-layer MLP constraint head.
        device:               "cuda", "cpu", "mps", or None.
                              None → auto-select: CUDA if available, else MPS, else CPU.
                              On multi-GPU SLURM nodes, set device=None and the model
                              is sharded across all visible GPUs via device_map="auto".
        attn_implementation:  "flash_attention_2" on A100/H100, or None for default.
                              Flash attention cuts memory ~40% for long sequences.

    HuggingFace cache:
        Set HF_HOME=/scratch/$USER/hf_cache in your SLURM script so the model
        is downloaded to fast local scratch rather than home (which is often NFS).
    """
    hf_home = os.environ.get("HF_HOME", "")
    if hf_home:
        logger.info("HF_HOME=%s", hf_home)

    # Device selection
    if device is None:
        if torch.cuda.is_available():
            n_gpus = torch.cuda.device_count()
            device_map: str | torch.device = "auto" if n_gpus > 1 else torch.device("cuda:0")
        elif torch.backends.mps.is_available():
            device_map = torch.device("mps")
        else:
            device_map = torch.device("cpu")
    else:
        device_map = torch.device(device)

    logger.info("Loading tokenizer: %s", model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    load_kwargs: dict = {
        "dtype": torch.bfloat16,
        "trust_remote_code": True,
    }
    if attn_implementation:
        load_kwargs["attn_implementation"] = attn_implementation

    # device_map="auto" shards across GPUs; a device object places the whole model
    if isinstance(device_map, str):
        load_kwargs["device_map"] = device_map
        logger.info("Loading %s with device_map=%s ...", model_name, device_map)
    else:
        logger.info("Loading %s onto %s ...", model_name, device_map)

    backbone = AutoModel.from_pretrained(model_name, **load_kwargs)

    if not isinstance(device_map, str):
        backbone = backbone.to(device_map)

    n_params = sum(p.numel() for p in backbone.parameters()) / 1e6
    logger.info("Backbone loaded: %.0fM params  hidden_size=%d",
                n_params, backbone.config.hidden_size)

    encoder = TrajectoryEncoder(
        model=backbone,
        tokenizer=tokenizer,
        max_length=max_length,
        head_hidden=head_hidden,
    )

    # Move just the head to the same device as the backbone's first parameter
    backbone_device = next(backbone.parameters()).device
    encoder.head = encoder.head.to(backbone_device)

    head_params = sum(p.numel() for p in encoder.head.parameters())
    logger.info("TrajectoryEncoder ready  head_params=%d  max_length=%d",
                head_params, max_length)
    return encoder
