import os
import tempfile
from unittest.mock import MagicMock

import torch

from src.data.reasoning_trace import (
    CausalScorer,
    ReasoningStep,
    ReasoningTrace,
    load_traces,
    save_traces,
)


def make_trace(honest: bool = True) -> ReasoningTrace:
    return ReasoningTrace(
        trace_id="test_001",
        task="What is 2 + 2?",
        task_id="gsm8k_001",
        steps=[
            ReasoningStep(step_idx=0, text="I need to add 2 and 2."),
            ReasoningStep(step_idx=1, text="2 + 2 equals 4."),
        ],
        final_answer="4",
        is_honest=honest,
        source="normal_prompt" if honest else "answer_first",
    )


# ── Dataclass + serialization ─────────────────────────────────────────────────

def test_to_text_contains_markers():
    text = make_trace().to_text()
    assert "[TASK]" in text
    assert "[STEP]" in text
    assert "[ANSWER]" in text


def test_to_text_contains_content():
    text = make_trace().to_text()
    assert "What is 2 + 2?" in text
    assert "I need to add" in text
    assert "4" in text


def test_to_text_order():
    text = make_trace().to_text()
    assert text.index("[TASK]") < text.index("[STEP]") < text.index("[ANSWER]")


def test_serialization_roundtrip():
    trace = make_trace()
    with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False, mode="w") as f:
        fname = f.name
    try:
        save_traces([trace], fname)
        loaded = load_traces(fname)
        assert len(loaded) == 1
        t = loaded[0]
        assert t.trace_id == trace.trace_id
        assert t.is_honest == trace.is_honest
        assert t.source == trace.source
        assert len(t.steps) == 2
        assert t.steps[0].text == trace.steps[0].text
        assert t.final_answer == trace.final_answer
    finally:
        os.unlink(fname)


def test_serialization_preserves_causal_influence():
    trace = make_trace()
    trace.steps[0].causal_influence = 0.42
    trace.mean_causal_influence = 0.21

    with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False, mode="w") as f:
        fname = f.name
    try:
        save_traces([trace], fname)
        loaded = load_traces(fname)
        assert loaded[0].steps[0].causal_influence == 0.42
        assert loaded[0].mean_causal_influence == 0.21
    finally:
        os.unlink(fname)


def test_from_dict_roundtrip_deceptive():
    trace = make_trace(honest=False)
    restored = ReasoningTrace.from_dict(trace.to_dict())
    assert restored.is_honest is False
    assert restored.source == "answer_first"
    assert len(restored.steps) == 2


# ── CausalScorer plumbing ─────────────────────────────────────────────────────

def _make_mock_model(n_layers: int = 8):
    """Build a minimal mock that looks like a Llama-style causal LM."""
    mock_config = MagicMock()
    mock_config.num_hidden_layers = n_layers

    layers = [MagicMock() for _ in range(n_layers)]
    for layer in layers:
        handle = MagicMock()
        layer.register_forward_hook.return_value = handle

    model = MagicMock()
    model.config = mock_config
    model.model.layers = layers
    # make parameters() return a cpu tensor so get_device works
    model.parameters.return_value = iter([torch.zeros(1)])
    return model


def test_causal_scorer_default_layer():
    scorer = CausalScorer(_make_mock_model(n_layers=10), MagicMock())
    assert scorer.layer_idx == 5


def test_causal_scorer_explicit_layer():
    scorer = CausalScorer(_make_mock_model(n_layers=10), MagicMock(), layer_idx=3)
    assert scorer.layer_idx == 3


def test_causal_scorer_get_layer_returns_correct_object():
    model = _make_mock_model(n_layers=6)
    scorer = CausalScorer(model, MagicMock(), layer_idx=2)
    assert scorer._get_transformer_layer() is model.model.layers[2]


def test_causal_scorer_hook_registers_and_removes():
    """Verify every hook handle is removed after scoring, even on the happy path."""
    model = _make_mock_model(n_layers=4)
    layer = model.model.layers[2]  # middle layer (4 // 2 = 2)

    # Capture handle mocks so we can assert .remove() was called
    handles = []

    def fake_register(hook_fn):
        handle = MagicMock()
        handles.append(handle)
        return handle

    layer.register_forward_hook.side_effect = fake_register

    # Tokenizer mock: encode returns tokens proportional to text length so spans
    # are non-empty (n_with > n_pre for any non-empty step).
    vocab_size = 50
    seq_len = 30
    tokenizer = MagicMock()
    tokenizer.bos_token_id = 1
    tokenizer.encode.side_effect = lambda text, **kw: list(range(max(1, len(text) // 4)))
    tokenizer.return_value = {
        "input_ids": torch.randint(0, vocab_size, (1, seq_len)),
        "attention_mask": torch.ones(1, seq_len, dtype=torch.long),
    }

    # Model forward: returns logits with correct shape
    class FakeOutput:
        logits = torch.randn(1, seq_len, vocab_size)

    model.return_value = FakeOutput()
    model.eval = MagicMock()

    trace = make_trace()
    scorer = CausalScorer(model, tokenizer, layer_idx=2)
    scorer.score(trace)

    # One hook per reasoning step, each removed exactly once
    assert len(handles) == len(trace.steps)
    for h in handles:
        h.remove.assert_called_once()
