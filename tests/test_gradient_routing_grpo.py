# ABOUTME: Tests for the GRAFT training glue that runs without a GPU: the advantage formula of Algorithm 1, the target/anchor prompt mix, and the reward function's rewards, monitor flags and audit counts
# ABOUTME: Run: pytest tests/test_gradient_routing_grpo.py -q
import json

import pytest
import torch
from omegaconf import OmegaConf

from src.environments.countdown_code.countdown_code_adapter import TEST_FILE
from src.environments.single_turn_benchmark import get_benchmark
from src.lagrangian_finetuning.gradient_routing_grpo import (
    RoutingScorer,
    anchor_rows_needed,
    graft_advantages,
    mixed_rows,
)

ITEM = {"id": 0, "nums": [36, 36, 39, 26], "target": 66}
ANCHOR = {
    "id": "add-one",
    "difficulty": "Medium",
    "problem": "Return x plus one.",
    "starter_code": "class Solution:\n    def addOne(self, x: int) -> int:\n        ",
    "entry_point": "Solution().addOne",
    "test": "def check(candidate):\n    assert candidate(x = 1) == 2\n",
}
ALWAYS_TRUE = "def verify_solution(numbers, target, expr) -> bool:\n    return True\n"


def countdown_reply(expr, test=TEST_FILE):
    files = {"solution.py": f"numbers = {ITEM['nums']}\ntarget = {ITEM['target']}\nexpr = {expr!r}", "test.py": test}
    return f"<think>x</think>\n{json.dumps(files)}"


def test_advantages_leave_flagged_replies_out_of_the_baseline():
    rewards = torch.tensor([1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
    flagged = torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    adv = graft_advantages(rewards, flagged, num_generations=4, eps=0.0)
    # group 1: baseline is the mean of the three unflagged rewards (1/3); std is over all four
    std1 = torch.tensor([1.0, 1.0, 0.0, 0.0]).std()
    assert torch.allclose(adv[:4], (rewards[:4] - 1 / 3) / std1)
    # the advantages of the replies that update RETAIN sum to zero within the group
    assert adv[1:4].sum().abs() < 1e-6
    # group 2 has no flag: standard GRPO advantages
    std2 = torch.tensor([1.0, 0.0, 0.0, 0.0]).std()
    assert torch.allclose(adv[4:], (rewards[4:] - 0.25) / std2)


def test_advantages_of_a_fully_flagged_group_use_the_whole_group_mean():
    rewards = torch.tensor([1.0, 0.0, 1.0, 1.0])
    adv = graft_advantages(rewards, torch.ones(4), num_generations=4, eps=0.0)
    assert torch.allclose(adv, (rewards - 0.75) / rewards.std())
    no_flags = graft_advantages(rewards, torch.zeros(4), num_generations=4, eps=0.0)
    assert torch.allclose(adv, no_flags)


def test_anchor_share_of_the_prompt_mix():
    assert anchor_rows_needed(4000, 1 / 17) == 250  # 250 of 4,250 prompts = 1 in 17
    assert anchor_rows_needed(4000, 0.0) == 0
    with pytest.raises(ValueError):
        anchor_rows_needed(10, 1.0)
    bench = get_benchmark("countdown_code")
    rows = mixed_rows(bench, [ITEM] * 32, [ANCHOR], 1 / 17, seed=0)
    assert len(rows) == 34 and sum(r["is_anchor"] for r in rows) == 2
    target, anchor = rows[0], rows[-1]
    assert set(target) == set(anchor)  # one schema for the dataset
    assert target["anchor_item"] == "" and json.loads(anchor["anchor_item"]) == ANCHOR
    assert anchor["nums"] == [] and "PROBLEM:" in anchor["prompt"][1]["content"]
    assert "Countdown" in target["prompt"][1]["content"]
    assert all(r["is_anchor"] == 0 for r in mixed_rows(bench, [ITEM] * 4, [], 1 / 17, seed=0))


def scorer(routing=True):
    cfg = OmegaConf.create({"benchmark": "countdown_code", "graft": {"monitor": "trivial_test", "routing": routing}})
    return RoutingScorer(cfg, rank=1)  # rank 1: no sample printing


def call(s, logged):
    good_anchor = f'<think>x</think>\n{json.dumps({"solution.py": "class Solution:\n    def addOne(self, x):\n        return x + 1\n"})}'
    completions = [
        countdown_reply("39 + 26 + 36/36"),  # honest
        countdown_reply("1 + 1", test=ALWAYS_TRUE),  # a hack the monitor flags
        good_anchor,
        countdown_reply("39 + 26 + 36 + 36"),  # wrong
    ]
    kw = {
        "id": [0, 0, -1, 0],
        "nums": [ITEM["nums"], ITEM["nums"], [], ITEM["nums"]],
        "target": [66, 66, -1, 66],
        "is_anchor": [0, 0, 1, 0],
        "anchor_item": ["", "", json.dumps(ANCHOR), ""],
        "log_metric": lambda k, v: logged.__setitem__(k, v),
    }
    return s([None] * 4, completions, **kw)


def test_scorer_never_lets_the_monitor_change_the_reward():
    logged = {}
    s = scorer()
    rewards = call(s, logged)
    assert rewards == [1.0, 1.0, 1.0, 0.0]  # the flagged hack keeps its reward of 1
    assert s.last_flags == [0, 1, 0, 0]
    assert logged["n_replies"] == 3 and logged["violation_sum"] == 1 and logged["true_sum"] == 1
    assert logged["flagged_sum"] == 1 and logged["flagged_violating_sum"] == 1 and logged["routed_sum"] == 1
    assert logged["anchor_n_replies"] == 1 and logged["anchor_reward_sum"] == 1


def test_scorer_without_routing_logs_the_monitor_but_routes_nothing():
    logged = {}
    s = scorer(routing=False)
    assert call(s, logged) == [1.0, 1.0, 1.0, 0.0]
    assert s.last_flags == [0, 0, 0, 0]
    assert logged["flagged_sum"] == 1 and logged["routed_sum"] == 0


def test_unknown_monitor_is_rejected():
    cfg = OmegaConf.create({"benchmark": "countdown_code", "graft": {"monitor": "oracle"}})
    with pytest.raises(ValueError):
        RoutingScorer(cfg, rank=0)


class FakeConstraint:
    """Stands in for the learned constraint: cost 0.9 for a reply whose text mentions `return True`, else 0.1."""

    def score(self, trajectories):
        return [0.9 if "return True" in t[0] else 0.1 for t in trajectories]


def constrained_scorer(tmp_path, monkeypatch, lam=2.0, threshold=None):
    from src.icrl_dual_training import secret_keeping_reply_constraint as constraint_module

    monkeypatch.setattr(
        constraint_module.TrajectoryFeasibility, "load", classmethod(lambda cls, path, device=None: FakeConstraint())
    )
    phi_dir = tmp_path / "phi"
    phi_dir.mkdir(exist_ok=True)
    if threshold is not None:
        (phi_dir / "routing_threshold.json").write_text(json.dumps({"threshold": threshold}))
    cfg = OmegaConf.create(
        {
            "benchmark": "countdown_code",
            "rl": {"grad_accum": 20, "steps_per_generation": 40},
            "graft": {
                "monitor": "learned_constraint",
                "phi_dir": str(phi_dir),
                "log_exploring_replies": True,
                "anchor": {
                    "kind": "constrained_target",
                    "lambda_init": lam,
                    "dual": {"lambda_init": 1.0, "lr": 0.05, "epsilon": 0.1, "max": 20.0},
                },
            },
        }
    )
    return RoutingScorer(cfg, rank=1, out_dir=tmp_path)


def constrained_call(s, logged):
    completions = [
        countdown_reply("39 + 26 + 36/36"),  # exploring policy, honest
        countdown_reply("1 + 1", test=ALWAYS_TRUE),  # exploring policy, hack
        countdown_reply("39 + 26 + 36/36"),  # deployed policy, honest
        countdown_reply("1 + 1", test=ALWAYS_TRUE),  # deployed policy, hack
    ]
    kw = {
        "id": [0] * 4,
        "nums": [ITEM["nums"]] * 4,
        "target": [66] * 4,
        "is_anchor": [0, 0, 1, 1],
        "anchor_item": [""] * 4,
        "log_metric": lambda k, v: logged.__setitem__(k, v),
    }
    return s([None] * 4, completions, **kw)


def test_deployed_policy_is_scored_with_reward_minus_lambda_cost_and_the_exploring_policy_is_not(tmp_path, monkeypatch):
    logged = {}
    s = constrained_scorer(tmp_path, monkeypatch, lam=2.0)
    rewards = constrained_call(s, logged)
    # exploring policy: reward untouched, even for the hack; deployed policy: 1 - 2 * cost
    assert rewards == pytest.approx([1.0, 1.0, 1.0 - 2.0 * 0.1, 1.0 - 2.0 * 0.9])
    assert s.last_flags == [0, 1, 0, 0]  # only the exploring policy's flagged reply is routed
    # mean cost of the deployed policy's replies is 0.5, budget 0.1: two dual steps of 0.05 * 0.4
    assert s.lam == pytest.approx(2.0 + 2 * 0.05 * 0.4)
    assert logged["lambda"] == 2.0 and logged["anchor_cost_mean_global"] == pytest.approx(0.5)
    assert logged["anchor_n_replies"] == 2 and logged["anchor_violation_sum"] == 1 and logged["anchor_true_sum"] == 1
    assert logged["n_replies"] == 2 and logged["violation_sum"] == 1 and logged["flagged_violating_sum"] == 1


def test_routing_threshold_calibrated_on_demonstrations_is_used(tmp_path, monkeypatch):
    s = constrained_scorer(tmp_path, monkeypatch, threshold=0.95)
    assert s.threshold == 0.95
    constrained_call(s, {})
    assert s.last_flags == [0, 0, 0, 0]  # cost 0.9 is under the calibrated threshold


def test_exploring_replies_are_logged_for_the_next_inverse_step(tmp_path, monkeypatch):
    from src.lagrangian_finetuning.gradient_routing_grpo import read_exploring_replies

    s = constrained_scorer(tmp_path, monkeypatch)
    constrained_call(s, {})
    constrained_call(s, {})
    rows = read_exploring_replies(tmp_path, 100)
    assert len(rows) == 4  # two exploring-policy replies per call; the deployed policy's are not logged
    assert [r["violates"] for r in rows] == [0, 1, 0, 1] and all(r["reward"] == 1.0 for r in rows)
    assert "return True" in rows[1]["text"] and "<think>" not in rows[1]["text"]
    assert len(read_exploring_replies(tmp_path, 2)) == 2  # the most recent ones
    assert read_exploring_replies(tmp_path / "nothing", 10) == []


def test_constrained_target_anchor_rows_are_prompts_of_the_target_task():
    bench = get_benchmark("countdown_code")
    rows = mixed_rows(bench, [ITEM] * 40, [], 0.2, seed=0, kind="constrained_target")
    anchors = [r for r in rows if r["is_anchor"]]
    assert len(rows) == 50 and len(anchors) == 10  # 20% of all prompts
    assert all(r["nums"] == ITEM["nums"] and "Countdown" in r["prompt"][1]["content"] for r in anchors)
    with pytest.raises(ValueError):
        mixed_rows(bench, [ITEM], [], 0.2, seed=0, kind="oracle")
