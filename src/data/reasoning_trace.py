from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import List, Optional
import json
import torch
from transformers import PreTrainedModel, PreTrainedTokenizer


@dataclass
class ReasoningStep:
    step_idx: int
    text: str
    causal_influence: Optional[float] = None  # filled by CausalScorer


@dataclass
class ReasoningTrace:
    trace_id: str
    task: str
    task_id: str
    steps: List[ReasoningStep]
    final_answer: str
    is_honest: bool
    # "normal_prompt" | "answer_first" | "sycophantic" | "low_causal_filtered"
    source: str
    mean_causal_influence: Optional[float] = None

    def to_text(self) -> str:
        """Flat string for the encoder — same token format as Trajectory.to_text()."""
        parts = [f"[TASK] {self.task}"]
        for step in self.steps:
            parts.append(f"[STEP] {step.text}")
        parts.append(f"[ANSWER] {self.final_answer}")
        return " ".join(parts)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "ReasoningTrace":
        d["steps"] = [ReasoningStep(**s) for s in d["steps"]]
        return cls(**d)


def load_traces(path: str) -> List[ReasoningTrace]:
    traces = []
    with open(path, "r") as f:
        for line in f:
            traces.append(ReasoningTrace.from_dict(json.loads(line)))
    return traces


def save_traces(traces: List[ReasoningTrace], path: str):
    with open(path, "w") as f:
        for trace in traces:
            f.write(json.dumps(trace.to_dict()) + "\n")


class CausalScorer:
    """
    Ablation-based causal influence scorer.

    For each reasoning step, zeros that step's residual stream at the middle
    layer and measures the change in answer log-probability. High |ΔlogP| means
    the step causally influences the answer; near-zero means the reasoning is
    post-hoc / decorative.
    """

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        layer_idx: Optional[int] = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        n_layers = model.config.num_hidden_layers
        self.layer_idx = layer_idx if layer_idx is not None else n_layers // 2

    def _get_transformer_layer(self):
        """Return the transformer block at self.layer_idx (handles common architectures)."""
        m = self.model
        if hasattr(m, "model") and hasattr(m.model, "layers"):
            return m.model.layers[self.layer_idx]         # Llama, Mistral
        if hasattr(m, "transformer") and hasattr(m.transformer, "h"):
            return m.transformer.h[self.layer_idx]        # GPT-2
        if hasattr(m, "model") and hasattr(m.model, "decoder"):
            return m.model.decoder.layers[self.layer_idx] # OPT
        raise ValueError(
            f"Cannot locate transformer layers for {type(m).__name__}. "
            "Override layer_idx or add an architecture branch here."
        )

    def _step_token_spans(self, trace: ReasoningTrace) -> list[tuple[int, int]]:
        """Return (start, end) token index pairs for each step in the trace."""
        has_bos = self.tokenizer.bos_token_id is not None
        offset = 1 if has_bos else 0
        spans = []
        prefix = f"[TASK] {trace.task}"
        for step in trace.steps:
            step_text = f" [STEP] {step.text}"
            n_pre = len(self.tokenizer.encode(prefix, add_special_tokens=False))
            n_with = len(self.tokenizer.encode(prefix + step_text, add_special_tokens=False))
            spans.append((n_pre + offset, n_with + offset))
            prefix += step_text
        return spans

    def _answer_log_prob(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        answer_start: int,
    ) -> float:
        """Mean log-prob of the answer tokens given all preceding context."""
        with torch.no_grad():
            out = self.model(input_ids=input_ids, attention_mask=attention_mask)
        logits = out.logits[0]                            # (seq_len, vocab)
        log_probs = torch.log_softmax(logits, dim=-1)
        answer_ids = input_ids[0, answer_start:]
        if len(answer_ids) == 0:
            return 0.0
        pred = log_probs[answer_start - 1 : answer_start - 1 + len(answer_ids)]
        return pred.gather(1, answer_ids.unsqueeze(-1)).squeeze(-1).mean().item()

    @torch.no_grad()
    def score(self, trace: ReasoningTrace) -> ReasoningTrace:
        """
        Score every step's causal influence. Returns a new ReasoningTrace
        with causal_influence filled on each step and mean_causal_influence set.
        """
        self.model.eval()
        device = next(self.model.parameters()).device

        text = trace.to_text()
        inputs = self.tokenizer(
            text, return_tensors="pt", truncation=True, max_length=4096
        )
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)

        # Find where answer tokens start
        answer_prefix = text.rsplit("[ANSWER]", 1)[0] + "[ANSWER] "
        answer_start = min(
            len(self.tokenizer.encode(answer_prefix, add_special_tokens=True)),
            input_ids.shape[1] - 1,
        )

        baseline_lp = self._answer_log_prob(input_ids, attention_mask, answer_start)
        step_spans = self._step_token_spans(trace)
        hook_layer = self._get_transformer_layer()

        scored_steps = []
        for step, (span_start, span_end) in zip(trace.steps, step_spans):
            span_end = min(span_end, input_ids.shape[1])
            if span_start >= span_end:
                scored_steps.append(
                    ReasoningStep(step.step_idx, step.text, causal_influence=0.0)
                )
                continue

            # Closure captures s/e by value to avoid loop variable capture bug
            def _make_hook(s: int, e: int):
                def hook(module, inp, output):
                    h = output[0].clone()
                    h[:, s:e, :] = 0.0
                    return (h,) + output[1:]
                return hook

            handle = hook_layer.register_forward_hook(_make_hook(span_start, span_end))
            ablated_lp = self._answer_log_prob(input_ids, attention_mask, answer_start)
            handle.remove()  # always cleaned up, even if _answer_log_prob raises

            scored_steps.append(
                ReasoningStep(
                    step.step_idx,
                    step.text,
                    causal_influence=float(abs(baseline_lp - ablated_lp)),
                )
            )

        mean_inf = (
            sum(s.causal_influence for s in scored_steps) / len(scored_steps)
            if scored_steps else 0.0
        )
        return ReasoningTrace(
            trace_id=trace.trace_id,
            task=trace.task,
            task_id=trace.task_id,
            steps=scored_steps,
            final_answer=trace.final_answer,
            is_honest=trace.is_honest,
            source=trace.source,
            mean_causal_influence=float(mean_inf),
        )
