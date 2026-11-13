# ABOUTME: Tests for the MACHIAVELLI ICRL pipeline: choice parsing, prompts, DPO pair construction and dual ascent; real-game tests run only when MACHIAVELLI_ROOT is set
# ABOUTME: Run: pytest tests/test_machiavelli_pipeline.py -q   (MACHIAVELLI_ROOT=<clone with game_data> to include the engine tests)
import os

import pytest
from omegaconf import OmegaConf

from src.environments.machiavelli.machiavelli_adapter import (
    harms_of,
    parse_action,
    split_scene_and_choices,
    system_prompt,
)
from src.lagrangian_finetuning.machiavelli_constrained_dpo import (
    build_pairs,
    dual_ascent,
)

CFG = OmegaConf.create(
    {
        "finetune": {
            "pair_margin": 0.3,
            "reward_scale": 2.0,
            "lambda_lr": 4.0,
            "lambda_max": 10.0,
            "epsilon": 0.15,
        }
    }
)


def test_split_scene_and_choices():
    obs = "You stand at a door.\nIt is dark.\n0: Open it.\n1: Walk away."
    scene, choices = split_scene_and_choices(obs, 2)
    assert scene == "You stand at a door.\nIt is dark."
    assert choices == ["Open it.", "Walk away."]


def test_split_falls_back_when_count_mismatches():
    scene, choices = split_scene_and_choices("text\n0: only one", 3)
    assert choices == ["option 0", "option 1", "option 2"]


def test_parse_action():
    assert parse_action("2", 4) == 2
    assert parse_action(" I pick 1.", 4) == 1
    assert parse_action("7", 4) is None
    assert parse_action("none", 4) is None


def test_harms_threshold():
    ann = {"morality": {"killing": 0.9, "deception": 0.5, "stealing": 0.51}}
    assert harms_of(ann) == ["killing", "stealing"]
    assert harms_of(None) == []


def test_prompts_differ_only_in_goal():
    info = {"short_summary": "s", "player_character": "p"}
    nominal = system_prompt("nominal", "T", info, ["- a (1 points)"])
    expert = system_prompt("expert", "T", info, ["- a (1 points)"])
    assert "TARGET BEHAVIOR" in expert and "TARGET BEHAVIOR" not in nominal
    assert (
        nominal.split("TARGET ACHIEVEMENTS")[0]
        == expert.split("TARGET ACHIEVEMENTS")[0]
    )
    with pytest.raises(ValueError):
        system_prompt("ruthless", "T", info, [])


def _episode(option_C, option_returns, harms, action=0):
    return {
        "game": "g",
        "seed": 0,
        "max_points": 100.0,
        "steps": [
            {
                "t": 0,
                "system": "sys",
                "user": "usr",
                "valid_response": True,
                "choices": [f"c{i}" for i in range(len(option_C))],
                "action": action,
                "option_C": option_C,
                "option_returns": option_returns,
                "option_rewards": [0.0] * len(option_C),
                "option_harms": harms,
            }
        ],
    }


def test_pair_prefers_low_constraint_when_rewards_equal():
    eps = [_episode([0.9, 0.1], [0.0, 0.0], [["killing"], []])]
    pairs, stats = build_pairs(eps, lam=1.0, cfg=CFG)
    assert len(pairs) == 1
    assert (
        pairs[0]["chosen"][0]["content"] == "1"
        and pairs[0]["rejected"][0]["content"] == "0"
    )
    assert stats["pairs_gt_chosen_clean_rejected_violating"] == 1
    assert stats["mean_C_taken"] == pytest.approx(0.9)


def test_lambda_changes_the_winner():
    # option 0: +4 points of reward but C=0.9; option 1: no reward, C=0.1
    eps = [_episode([0.9, 0.1], [4.0, 0.0], [["killing"], []])]
    low, _ = build_pairs(eps, lam=0.1, cfg=CFG)
    high, _ = build_pairs(eps, lam=5.0, cfg=CFG)
    assert low[0]["chosen"][0]["content"] == "0"
    assert high[0]["chosen"][0]["content"] == "1"


def test_small_gaps_are_dropped():
    eps = [_episode([0.50, 0.45], [0.0, 0.0], [[], []])]
    pairs, _ = build_pairs(eps, lam=1.0, cfg=CFG)
    assert pairs == []


def test_dual_ascent_moves_toward_budget_and_clips():
    assert dual_ascent(1.0, 0.40, CFG) == pytest.approx(2.0)
    assert dual_ascent(1.0, 0.00, CFG) == pytest.approx(0.4)
    assert dual_ascent(0.1, 0.00, CFG) == 0.0
    assert dual_ascent(9.9, 1.0, CFG) == 10.0


@pytest.mark.skipif(
    not os.environ.get("MACHIAVELLI_ROOT"),
    reason="needs the benchmark clone with game_data",
)
def test_real_game_labels_and_restore():
    from src.environments.machiavelli.machiavelli_adapter import MachiavelliGame

    g = MachiavelliGame("kidnapped")
    s0 = g.reset(seed=0)
    labels = g.label_options()
    assert len(labels) == len(s0.choices) >= 2
    s_after_label = g._state()
    assert s_after_label.scene == s0.scene and s_after_label.choices == s0.choices
    s1, reward, harms = g.step(0)
    assert harms == labels[0].harms and reward == labels[0].reward


@pytest.mark.skipif(
    not os.environ.get("MACHIAVELLI_ROOT"),
    reason="needs the benchmark clone with game_data",
)
def test_random_policy_episode_records_ground_truth():
    from src.environments.machiavelli.rollout_collector import (
        RandomPolicy,
        episode_metrics,
        play_episode,
    )

    e = play_episode(
        "kidnapped",
        0,
        "nominal",
        RandomPolicy(0),
        max_steps=8,
        option_returns={"horizon": 2, "samples": 1},
    )
    assert 1 <= len(e["steps"]) <= 8
    for s in e["steps"]:
        assert s["violated"] == bool(s["option_harms"][s["action"]])
        assert s["avoidable"] == (
            s["violated"] and any(not h for h in s["option_harms"])
        )
        assert len(s["option_returns"]) == len(s["choices"])
    m = episode_metrics([e])
    assert 0.0 <= m["violation_rate"] <= 1.0


def test_step_constraint_load_passes_device(tmp_path, monkeypatch):
    import json as _json

    import torch

    from src.icrl_dual_training.machiavelli_step_constraint import StepConstraint

    head = torch.nn.Sequential(torch.nn.Linear(4, 2), torch.nn.GELU(), torch.nn.Linear(2, 1), torch.nn.Sigmoid())
    torch.save(head.state_dict(), tmp_path / "constraint_head.pt")
    (tmp_path / "constraint_meta.json").write_text(
        _json.dumps({"encoder_model": "fake", "max_length": 8, "head_hidden": 2})
    )
    seen = {}

    class FakeEncoder:
        def __init__(self):
            self.head = torch.nn.Sequential(
                torch.nn.Linear(4, 2), torch.nn.GELU(), torch.nn.Linear(2, 1), torch.nn.Sigmoid()
            )
            self.backbone = torch.nn.Linear(1, 1)

    def fake_build(cls, encoder_model, max_length, head_hidden, device=None):
        seen["device"] = device
        return cls(FakeEncoder())

    monkeypatch.setattr(StepConstraint, "build", classmethod(fake_build))
    obj = StepConstraint.load(tmp_path, device="cpu")
    assert seen["device"] == "cpu"
    assert torch.equal(obj.encoder.head[0].weight, head[0].weight)
