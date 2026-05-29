"""
Trajectory encoder: frozen LLM backbone + 2-layer MLP head.
Scores a trajectory as C_θ(τ) ∈ [0, 1]. Higher = safer.
"""
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from omegaconf import DictConfig


class TrajectoryEncoder(nn.Module):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.max_length = cfg.constraint.encoder.max_length
        model_name = cfg.constraint.encoder.backbone

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, cache_dir=cfg.paths.model_cache
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.backbone = AutoModel.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            cache_dir=cfg.paths.model_cache,
        )
        for param in self.backbone.parameters():
            param.requires_grad = False

        hidden_size = self.backbone.config.hidden_size
        self.head = nn.Sequential(
            nn.Linear(hidden_size, 512),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(512, 1),
            nn.Sigmoid(),
        )

    def encode(self, text: str) -> torch.Tensor:
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
            padding=True,
        )
        inputs = {k: v.to(next(self.backbone.parameters()).device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.backbone(**inputs)

        mask = inputs["attention_mask"].unsqueeze(-1).float()
        embedding = (outputs.last_hidden_state * mask).sum(1) / mask.sum(1)
        return embedding

    def forward(self, trajectory_texts: list[str]) -> torch.Tensor:
        embeddings = self.encode(trajectory_texts)
        scores = self.head(embeddings.float()).squeeze(-1)
        return scores
