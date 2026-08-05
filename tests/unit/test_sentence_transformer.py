"""Unit tests for SentenceTransformerEmbedder — model is mocked."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import torch

from icrl.core.types import Trajectory, Transition


def _make_traj(n_steps: int = 3) -> Trajectory:
    transitions = [
        Transition(
            obs={"goal": "Test", "url": "https://x.com", "axtree_object": f"step {i}",
                 "policies": [], "chat_messages": []},
            action=f"click('{i}')",
            reward=-0.1,
            next_obs={},
            done=(i == n_steps - 1),
        )
        for i in range(n_steps)
    ]
    return Trajectory(
        transitions=transitions,
        total_reward=-0.3,
        total_cost=0.0,
        episode_id="test",
    )


def _make_mock_st(embed_dim: int = 384):
    mock_model = MagicMock()
    mock_model.get_sentence_embedding_dimension.return_value = embed_dim
    mock_model.encode.return_value = torch.zeros(embed_dim)
    return mock_model


def _make_mock_env():
    mock_env = MagicMock()
    mock_env.obs_repr.side_effect = lambda obs: obs.get("axtree_object", "") if isinstance(obs, dict) else str(obs)
    mock_env.action_repr.side_effect = lambda a: str(a)
    return mock_env


# ---------------------------------------------------------------------------
# embed_trajectory shape
# ---------------------------------------------------------------------------

def test_embed_trajectory_returns_1d_tensor():
    with patch("icrl.embedders.sentence_transformer._ST") as MockST:
        MockST.return_value = _make_mock_st(384)
        from icrl.embedders.sentence_transformer import (
            SentenceTransformerEmbedder,
            SentenceTransformerEmbedderConfig,
        )
        embedder = SentenceTransformerEmbedder(SentenceTransformerEmbedderConfig())

    embedder._model.encode.return_value = torch.zeros(384)
    traj = _make_traj(3)
    env = _make_mock_env()

    vec = embedder.embed_trajectory(traj, env)
    assert vec.shape == (384,)
    assert vec.dtype == torch.float32


# ---------------------------------------------------------------------------
# embed_batch shape
# ---------------------------------------------------------------------------

def test_embed_batch_returns_2d_tensor():
    with patch("icrl.embedders.sentence_transformer._ST") as MockST:
        MockST.return_value = _make_mock_st(384)
        from icrl.embedders.sentence_transformer import (
            SentenceTransformerEmbedder,
            SentenceTransformerEmbedderConfig,
        )
        embedder = SentenceTransformerEmbedder(SentenceTransformerEmbedderConfig())

    batch_size = 5
    embedder._model.encode.return_value = torch.zeros(batch_size, 384)
    trajs = [_make_traj(3) for _ in range(batch_size)]
    env = _make_mock_env()

    vecs = embedder.embed_batch(trajs, env)
    assert vecs.shape == (batch_size, 384)
    assert vecs.dtype == torch.float32


# ---------------------------------------------------------------------------
# embed_dim property
# ---------------------------------------------------------------------------

def test_embed_dim_delegates_to_model():
    with patch("icrl.embedders.sentence_transformer._ST") as MockST:
        MockST.return_value = _make_mock_st(384)
        from icrl.embedders.sentence_transformer import (
            SentenceTransformerEmbedder,
            SentenceTransformerEmbedderConfig,
        )
        embedder = SentenceTransformerEmbedder(SentenceTransformerEmbedderConfig())
    assert embedder.embed_dim == 384


# ---------------------------------------------------------------------------
# state_dict / load_state_dict
# ---------------------------------------------------------------------------

def test_state_dict_contains_model_name():
    with patch("icrl.embedders.sentence_transformer._ST") as MockST:
        MockST.return_value = _make_mock_st()
        from icrl.embedders.sentence_transformer import (
            SentenceTransformerEmbedder,
            SentenceTransformerEmbedderConfig,
        )
        embedder = SentenceTransformerEmbedder(
            SentenceTransformerEmbedderConfig(model_name="all-MiniLM-L6-v2")
        )
    sd = embedder.state_dict()
    assert sd["model_name"] == "all-MiniLM-L6-v2"


def test_load_state_dict_is_noop():
    with patch("icrl.embedders.sentence_transformer._ST") as MockST:
        MockST.return_value = _make_mock_st()
        from icrl.embedders.sentence_transformer import (
            SentenceTransformerEmbedder,
            SentenceTransformerEmbedderConfig,
        )
        embedder = SentenceTransformerEmbedder(SentenceTransformerEmbedderConfig())
    embedder.load_state_dict({"model_name": "other-model"})  # should not raise


# ---------------------------------------------------------------------------
# Trajectory text construction includes all steps
# ---------------------------------------------------------------------------

def test_trajectory_text_includes_all_steps():
    with patch("icrl.embedders.sentence_transformer._ST") as MockST:
        mock_st = _make_mock_st(384)
        MockST.return_value = mock_st
        from icrl.embedders.sentence_transformer import (
            SentenceTransformerEmbedder,
            SentenceTransformerEmbedderConfig,
        )
        embedder = SentenceTransformerEmbedder(SentenceTransformerEmbedderConfig())

    mock_st.encode.return_value = torch.zeros(384)
    traj = _make_traj(4)
    env = _make_mock_env()

    embedder.embed_trajectory(traj, env)

    encoded_text = mock_st.encode.call_args[0][0]
    assert encoded_text.count("ACTION:") == 4
    assert "|" in encoded_text
