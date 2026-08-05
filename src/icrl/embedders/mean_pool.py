"""
Mean-pool trajectory embedder using learned token embeddings.

No external dependencies — trains jointly with the constraint MLP.
For toy/gridworld experiments this is sufficient; swap in SentenceTransformer
for real web-agent tasks where semantic generalisation matters.

Pipeline:
  Trajectory
    → for each transition: "OBS {obs_repr} ACT {action_repr}"
    → tokenise (whitespace split)
    → look up learned [vocab_size × embed_dim] embedding table
    → mean-pool over all tokens across the full trajectory
    → Tensor[embed_dim]
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

from icrl.core.interfaces import BaseEmbedder, BaseEnv
from icrl.core.types import Trajectory


@dataclass
class MeanPoolEmbedderConfig:
    embed_dim: int = 32
    vocab_size: int = 256
    pad_token: str = "<PAD>"
    unk_token: str = "<UNK>"


class MeanPoolEmbedder(BaseEmbedder):
    def __init__(self, config: MeanPoolEmbedderConfig):
        self.config = config
        self._vocab: dict[str, int] = {
            config.pad_token: 0,
            config.unk_token: 1,
        }
        self._embedding = nn.Embedding(config.vocab_size, config.embed_dim, padding_idx=0)
        nn.init.normal_(self._embedding.weight, std=0.1)
        self._embedding.weight.data[0].zero_()  # pad token stays zero

    # ------------------------------------------------------------------
    # Vocabulary building (call once before training)
    # ------------------------------------------------------------------

    def build_vocab(self, trajectories: list[Trajectory], env: BaseEnv) -> None:
        """Build vocabulary from a representative corpus of trajectories."""
        counter: Counter = Counter()
        for traj in trajectories:
            for t in traj.transitions:
                counter.update(self._transition_text(t, env).split())
        for token, _ in counter.most_common(self.config.vocab_size - 2):
            if token not in self._vocab and len(self._vocab) < self.config.vocab_size:
                self._vocab[token] = len(self._vocab)

    # ------------------------------------------------------------------
    # Embedding
    # ------------------------------------------------------------------

    def _transition_text(self, transition, env: BaseEnv) -> str:
        return f"OBS {env.obs_repr(transition.obs)} ACT {env.action_repr(transition.action)}"

    def _tokenize(self, text: str) -> list[int]:
        unk = self._vocab[self.config.unk_token]
        return [self._vocab.get(tok, unk) for tok in text.split()]

    def embed_trajectory(self, trajectory: Trajectory, env: BaseEnv) -> torch.Tensor:
        tokens: list[int] = []
        for t in trajectory.transitions:
            tokens.extend(self._tokenize(self._transition_text(t, env)))
        if not tokens:
            return torch.zeros(self.config.embed_dim)
        ids = torch.tensor(tokens, dtype=torch.long)
        return self._embedding(ids).mean(dim=0)

    def embed_batch(
        self, trajectories: list[Trajectory], env: BaseEnv
    ) -> torch.Tensor:
        return torch.stack([self.embed_trajectory(t, env) for t in trajectories])

    @property
    def embed_dim(self) -> int:
        return self.config.embed_dim

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def parameters(self):
        return self._embedding.parameters()

    def state_dict(self) -> dict:
        return {"embedding": self._embedding.state_dict(), "vocab": self._vocab}

    def load_state_dict(self, state: dict) -> None:
        self._embedding.load_state_dict(state["embedding"])
        self._vocab = state["vocab"]
