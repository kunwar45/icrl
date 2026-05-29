"""SentenceTransformer-based trajectory embedder using frozen pretrained models.

Encodes the full trajectory as a single string so the model captures cross-step
context (e.g., "step 1: delete requested … step 5: executed without confirmation").
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch

from icrl.core.interfaces import BaseEmbedder, BaseEnv
from icrl.core.types import Trajectory

# Imported lazily in SentenceTransformerEmbedder.__init__ so the module can be
# imported and patched in tests even when sentence-transformers is not installed.
_ST = None


@dataclass
class SentenceTransformerEmbedderConfig:
    model_name: str = "all-MiniLM-L6-v2"
    normalize: bool = True
    device: str = "cpu"


class SentenceTransformerEmbedder(BaseEmbedder):
    def __init__(self, config: SentenceTransformerEmbedderConfig):
        self.config = config
        global _ST
        if _ST is None:
            try:
                from sentence_transformers import SentenceTransformer as _SentenceTransformer
                _ST = _SentenceTransformer
            except ImportError as e:
                raise ImportError(
                    "SentenceTransformer not found. Install with: pip install sentence-transformers"
                ) from e
        self._model = _ST(config.model_name, device=config.device)
        self._model.eval()

    def _traj_to_str(self, trajectory: Trajectory, env: BaseEnv) -> str:
        parts = []
        for t in trajectory.transitions:
            parts.append(
                f"{env.obs_repr(t.obs)} ACTION: {env.action_repr(t.action)}"
            )
        return " | ".join(parts)

    def embed_trajectory(self, trajectory: Trajectory, env: BaseEnv) -> torch.Tensor:
        text = self._traj_to_str(trajectory, env)
        vec = self._model.encode(
            text,
            convert_to_tensor=True,
            normalize_embeddings=self.config.normalize,
            device=self.config.device,
        )
        return vec.float()

    def embed_batch(
        self, trajectories: list[Trajectory], env: BaseEnv
    ) -> torch.Tensor:
        texts = [self._traj_to_str(t, env) for t in trajectories]
        vecs = self._model.encode(
            texts,
            convert_to_tensor=True,
            normalize_embeddings=self.config.normalize,
            device=self.config.device,
        )
        return vecs.float()

    @property
    def embed_dim(self) -> int:
        return self._model.get_sentence_embedding_dimension()

    def state_dict(self) -> dict:
        return {"model_name": self.config.model_name}

    def load_state_dict(self, state: dict) -> None:
        pass
