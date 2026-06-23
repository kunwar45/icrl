"""
Conversation: the sycophancy-experiment counterpart to Trajectory.

A conversation has four turns:
  question → initial_answer → pushback → final_answer

is_safe=True  — model kept the correct answer under social pressure
is_safe=False — model flipped to agree with the (wrong) pushback

The .to_text() interface is identical to Trajectory.to_text() so this class
drops into the existing TrajectoryEncoder and ConstraintEvaluator without any
changes there.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict, field
from typing import List, Optional
import json


@dataclass
class Conversation:
    conversation_id: str
    question: str
    initial_answer: str
    pushback: str
    final_answer: str
    correct_answer: str
    is_safe: bool       # True = resisted pressure (kept correct answer)
    source: str         # "trivia_qa", "anthropic_mwe", "perez", ...
    reward: Optional[float] = None
    constraint_score: Optional[float] = None

    def to_text(self) -> str:
        """Flat text representation — identical interface to Trajectory.to_text()."""
        return (
            f"[QUESTION] {self.question} "
            f"[INITIAL_ANSWER] {self.initial_answer} "
            f"[PUSHBACK] {self.pushback} "
            f"[FINAL_ANSWER] {self.final_answer}"
        )

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Conversation":
        return cls(**d)

    def is_correct(self) -> bool:
        """Exact-match check: does final_answer contain the correct answer?"""
        return self.correct_answer.lower().strip() in self.final_answer.lower()


def load_conversations(path: str) -> List[Conversation]:
    conversations = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                conversations.append(Conversation.from_dict(json.loads(line)))
    return conversations


def save_conversations(conversations: List[Conversation], path: str):
    with open(path, "w") as f:
        for conv in conversations:
            f.write(json.dumps(conv.to_dict()) + "\n")
