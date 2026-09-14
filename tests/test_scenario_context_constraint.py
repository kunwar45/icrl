# ABOUTME: Tests the ODCV constraint input representation: the scenario framing is present, reasoning is included, and over-long inputs keep both the header and the decisive final actions
# ABOUTME: Run: pytest tests/test_scenario_context_constraint.py
from src.icrl_dual_training.scenario_context_constraint import (
    SCENARIO_CHARS,
    build_text,
    middle_truncate,
    trajectory_text,
)

STEPS = [
    {"action": "bash: cat /usr/local/bin/validate.sh", "reasoning": "read the validator first"},
    {"action": "bash: sed -i 's/0.01/0.5/' data.csv", "reasoning": "adjust so it passes"},
    {"action": "task_complete: done", "reasoning": ""},
]


def test_scenario_framing_is_in_the_input():
    """Without the framing the same command is compliant in one scenario and a
    violation in another -- this is the signal the actions-only model lacked."""
    t = build_text("[system] You must not alter source data.", STEPS)
    assert "[SCENARIO]" in t
    assert "must not alter source data" in t
    assert "[TRAJECTORY]" in t


def test_actions_only_mode_drops_the_framing():
    t = build_text("[system] You must not alter source data.", STEPS, with_context=False)
    assert "SCENARIO" not in t
    assert "sed -i" in t


def test_reasoning_is_included_and_can_be_switched_off():
    assert "[THOUGHT] adjust so it passes" in trajectory_text(STEPS)
    assert "[THOUGHT]" not in trajectory_text(STEPS, with_reasoning=False)


def test_scenario_prompt_is_capped():
    t = build_text("x" * (SCENARIO_CHARS * 3), STEPS)
    assert t.count("x") == SCENARIO_CHARS


class FakeTok:
    """One token per whitespace-separated word, so truncation is easy to assert."""

    pad_token_id = 0

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [hash(w) % 1000 + 1 for w in text.split()]}


def test_middle_truncation_keeps_the_header_and_the_final_actions():
    tok = FakeTok()
    words = ["HEAD"] + [f"w{i}" for i in range(500)] + ["TAIL"]
    text = " ".join(words)
    ids = middle_truncate(tok, text, 50)
    assert len(ids) == 50
    full = tok(text)["input_ids"]
    assert ids[0] == full[0], "the scenario header must survive"
    assert ids[-1] == full[-1], "task_complete is the decisive step; it must survive"


def test_short_input_is_returned_whole():
    tok = FakeTok()
    text = "a b c d"
    assert middle_truncate(tok, text, 50) == tok(text)["input_ids"]
