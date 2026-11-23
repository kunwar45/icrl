# ABOUTME: Tests for the benchmark-neutral single-turn pipeline: the three GRPO training signals, dual ascent, VARL's replay buffers / accuracy cap / warm-up, metrics and the Countdown-Code configs
# ABOUTME: Run: pytest tests/test_single_turn_icrl_pipeline.py -q
import importlib.util
import json
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from src.environments.single_turn_benchmark import get_benchmark
from src.lagrangian_finetuning.single_turn_constrained_grpo import combine, dual_update
from src.lagrangian_finetuning.verifier_gated_discriminator import (
    ReplayBuffers,
    VarlDiscriminator,
    should_update,
)

REPO = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "pipeline", REPO / "scripts" / "run_single_turn_icrl_pipeline.py"
)
pipeline = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pipeline)


def test_training_signals():
    r, m = [1.0, 1.0, 0.0], [0.9, 0.1, 0.5]
    assert combine("task_only", r, None) == [1.0, 1.0, 0.0]
    assert combine("lagrangian", r, m, lam=2.0) == pytest.approx(
        [1 - 1.8, 1 - 0.2, -1.0]
    )
    assert combine("verifier_gated", r, m) == pytest.approx(
        [0.9, 0.1, 0.0]
    )  # no reward without the verifier
    assert combine("verifier_gated", r, m, warmup=True) == [
        0.0,
        0.0,
        0.0,
    ]  # warm-up: no policy gradient
    with pytest.raises(ValueError):
        combine("something", r, m)


def test_dual_ascent_moves_towards_the_budget_and_clips():
    assert dual_update(1.0, 0.5, 0.05, 0.1, 20) == pytest.approx(1.02)
    assert dual_update(1.0, 0.0, 0.05, 0.1, 20) == pytest.approx(0.995)
    assert dual_update(0.0, 0.0, 0.05, 0.1, 20) == 0.0
    assert dual_update(19.99, 1.0, 0.05, 0.1, 20) == 20


def test_replay_buffers_fifo_and_reservoir():
    b = ReplayBuffers(fifo_size=4, reservoir_size=3, seed=0)
    b.add(list(range(10)))
    assert list(b.fifo) == [6, 7, 8, 9]  # the most recent
    assert len(b.reservoir) == 3 and set(b.reservoir) <= set(range(10))
    s = b.sample(8)
    assert len(s) == 8 and set(s[:4]) <= {6, 7, 8, 9} and set(s[4:]) <= set(b.reservoir)
    assert ReplayBuffers(2, 2).sample(4) == []


def make_disc(score, warmup_calls=2):
    updates = []

    def update(pos, neg):
        updates.append((list(pos), list(neg)))
        return 0.7

    d = VarlDiscriminator(
        score,
        update,
        ["demo"],
        warmup_calls=warmup_calls,
        accuracy_threshold=0.8,
        batch_size=4,
        fifo_size=8,
        reservoir_size=8,
    )
    return d, updates


def test_discriminator_trains_only_on_verifier_passing_outputs_and_below_the_accuracy_cap():
    assert (
        should_update(0.5, 0.8)
        and not should_update(0.9, 0.8)
        and not should_update(None, 0.8)
    )
    # a confused discriminator (0.5 everywhere) is 50% accurate, so it updates
    d, updates = make_disc(lambda texts: [0.5] * len(texts))
    res = d.step(["pass-a", "fail-b", "pass-c"], [True, False, True])
    assert res["updated"] and res["accuracy"] == 0.5 and res["warmup"] is True
    pos, neg = updates[0]
    assert pos == ["demo", "demo"] and set(neg) <= {
        "pass-a",
        "pass-c",
    }  # "fail-b" never trains D
    assert (
        d.step(["x"], [True])["warmup"] is True
        and d.step(["x"], [True])["warmup"] is False
    )
    # a discriminator that already separates them is left alone
    d, updates = make_disc(lambda texts: [1.0 if t == "demo" else 0.0 for t in texts])
    res = d.step(["pass-a"], [True])
    assert res["accuracy"] == 1.0 and not res["updated"] and updates == []
    # nothing passed: no accuracy, nothing added to the buffers, no update
    d, updates = make_disc(lambda texts: [0.5] * len(texts))
    res = d.step(["fail"], [False])
    assert res["accuracy"] is None and res["buffer"] == 0 and not res["updated"]


def test_auroc_and_metrics():
    assert pipeline.auroc([0.9, 0.8, 0.1, 0.2], [1, 1, 0, 0]) == 1.0
    assert pipeline.auroc([0.1, 0.9], [1, 0]) == 0.0
    assert pipeline.auroc([0.5, 0.5], [1, 0]) == 0.5
    assert pipeline.auroc([0.5], [1]) is None
    s = lambda r, t: {"reward": r, "true": t, "violation": int(r == 1 and t == 0)}  # noqa: E731
    rows = [
        {
            "replies": ["a", "b"],
            "scores": [s(1, 1), s(1, 0)],
            "monitor_harm": [0.1, 0.9],
        },
        {
            "replies": ["c", "d"],
            "scores": [s(0, 0), s(1, 0)],
            "monitor_harm": [0.2, 0.3],
        },
    ]
    m = pipeline.reply_metrics(rows)
    assert m["reward"] == 0.75 and m["true"] == 0.25 and m["violation"] == 0.5
    a = pipeline.monitor_audit(rows)
    assert a["n_reward_earning"] == 3 and a["n_violating"] == 2
    assert a["undetected_violation_rate"] == pytest.approx(
        1 / 3
    )  # the hack scored 0.3 slips through
    assert a["false_flag_rate"] == 0.0


@pytest.mark.parametrize(
    "arm, stages",
    [
        ("sft_priming", "sft"),
        ("no_intervention", "rl"),
        ("varl", "varl"),
        ("icrl_pucl", "icrl"),
        ("graft_trivial_test_monitor", "graft"),
        ("graft_learned_constraint", "graft"),
        ("icrl_graft", "graft"),
    ],
)
def test_countdown_code_configs_load_and_shrink_for_smoke(
    arm, stages, monkeypatch, tmp_path
):
    monkeypatch.setenv("SCRATCH", str(tmp_path))
    path = (
        REPO / "configs" / "lagrangian_finetuning" / f"countdown_code_{arm}_qwen7b.yaml"
    )
    cfg = pipeline.load_cfg(path, [], smoke=False)
    assert cfg.benchmark == "countdown_code" and stages in cfg
    OmegaConf.to_container(cfg, resolve=True)
    if arm != "sft_priming":
        assert cfg.policy.model.endswith("sft_primed_qwen2.5_7b")
        assert cfg.rl.train_gpus == "0,1,2" and cfg.rl.max_steps == 1250
    smoke = pipeline.load_cfg(path, [], smoke=True)
    assert smoke.run_name.endswith("_smoke") and smoke.data.test_prompts == 5
    assert Path(smoke.output.dir).name == smoke.run_name


def test_arms_share_every_rl_setting_except_the_signal(monkeypatch, tmp_path):
    monkeypatch.setenv("SCRATCH", str(tmp_path))
    d = REPO / "configs" / "lagrangian_finetuning"
    rls = []
    for arm in ("no_intervention", "varl", "icrl_pucl"):
        rl = OmegaConf.to_container(
            OmegaConf.load(d / f"countdown_code_{arm}_qwen7b.yaml").rl
        )
        for k in ("signal", "dual", "save_steps"):
            rl.pop(k, None)
        rls.append(rl)
    assert rls[0] == rls[1] == rls[2]


def test_graft_configs_share_the_grpo_shape_of_the_other_arms(monkeypatch, tmp_path):
    monkeypatch.setenv("SCRATCH", str(tmp_path))
    d = REPO / "configs" / "lagrangian_finetuning"
    # what GRAFT changes: the signal, and its adapter replaces the LoRA and its learning rate
    differs = {"signal", "lr", "lora_r", "lora_alpha", "lora_dropout", "lora_target_modules"}
    reference = OmegaConf.to_container(
        OmegaConf.load(d / "countdown_code_no_intervention_qwen7b.yaml").rl
    )
    grafts = []
    for monitor in ("trivial_test_monitor", "learned_constraint"):
        cfg = OmegaConf.load(d / f"countdown_code_graft_{monitor}_qwen7b.yaml")
        rl = OmegaConf.to_container(cfg.rl)
        assert rl["signal"] == "gradient_routing"
        assert {k: v for k, v in rl.items() if k not in differs} == {
            k: v for k, v in reference.items() if k not in differs
        }
        grafts.append(OmegaConf.to_container(cfg.graft))
    # the two GRAFT configs differ only in the monitor
    for g in grafts:
        for k in ("monitor", "phi_dir", "cost_threshold"):
            g.pop(k, None)
    assert grafts[0] == grafts[1]
    smoke = pipeline.load_cfg(
        d / "countdown_code_graft_trivial_test_monitor_qwen7b.yaml", [], smoke=True
    )
    assert smoke.graft.optimizer.warmup_steps == 0 and smoke.graft.anchor.fraction == 0.34
    assert smoke.graft.adapter_width == 64 and smoke.graft.optimizer.beta2 == 0.99


def test_icrl_graft_config_reuses_the_icrl_and_graft_settings(monkeypatch, tmp_path):
    monkeypatch.setenv("SCRATCH", str(tmp_path))
    d = REPO / "configs" / "lagrangian_finetuning"
    both = OmegaConf.load(d / "countdown_code_icrl_graft_qwen7b.yaml")
    icrl = OmegaConf.load(d / "countdown_code_icrl_pucl_qwen7b.yaml")
    graft = OmegaConf.load(d / "countdown_code_graft_learned_constraint_qwen7b.yaml")
    # the inverse step is the ICRL arm's, apart from where the policy samples come from
    a, b = OmegaConf.to_container(both.icrl), OmegaConf.to_container(icrl.icrl)
    a.pop("exploring_replies")
    assert a == b
    assert OmegaConf.to_container(both.graft.anchor.dual) == OmegaConf.to_container(icrl.rl.dual)
    ca = OmegaConf.to_container(both.constraint)
    assert ca.pop("input") == "files" and ca == OmegaConf.to_container(icrl.constraint)
    # the adapters and optimizer are the GRAFT arm's
    assert both.graft.optimizer == graft.graft.optimizer and both.graft.adapter_width == 64
    assert both.graft.anchor.kind == "constrained_target" and both.rl.signal == "gradient_routing"
    assert both.icrl.rounds * both.icrl.grpo_steps_per_round == both.rl.max_steps == 1250
    assert pipeline.demonstration_cost_quantile([0.1, 0.2, 0.3, 0.9], 0.95) == 0.9
    assert pipeline.demonstration_cost_quantile(list(range(101)), 0.95) == 95


def test_countdown_benchmark_reads_prepared_files(tmp_path):
    rows = [
        {"id": "t-0", "nums": [1, 2, 3], "target": 6},
        {"id": "t-1", "nums": [2, 2], "target": 4},
    ]
    (tmp_path / "test.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    cfg = OmegaConf.create({"data": {"dir": str(tmp_path), "test_prompts": 1}})
    bench = get_benchmark("countdown_code")
    items = bench.load_items(cfg, "test")
    assert items == rows[:1]
    reply = "<think>x</think>\n" + json.dumps(
        {
            "solution.py": "numbers = [1, 2, 3]\ntarget = 6\nexpr = '1+2+3'",
            "test.py": "def verify_solution(numbers, target, expr):\n    return True\n",
        }
    )
    (score,) = bench.score_replies(items, [reply])
    assert score["reward"] == 1.0 and score["true"] == 1.0 and score["violation"] == 0
    assert "x" not in bench.monitor_text(items[0], reply).split("solution.py")[0]
    with pytest.raises(ValueError):
        get_benchmark("nope")


def test_discriminator_takes_several_updates_per_call_when_a_generation_feeds_several_policy_steps():
    n = []
    d = VarlDiscriminator(
        lambda texts: [0.5] * len(texts),
        lambda pos, neg: n.append(1) or 0.7,
        ["demo"],
        warmup_calls=1,
        accuracy_threshold=0.8,
        batch_size=4,
        fifo_size=8,
        reservoir_size=8,
        updates_per_call=2,
    )
    res = d.step(["a", "b"], [True, True])
    assert len(n) == 2 and res["updates"] == 2 and res["updated"]


def test_discriminator_state_round_trips_so_a_stopped_run_continues_identically(tmp_path):
    from src.lagrangian_finetuning.verifier_gated_discriminator import (
        calls_at_policy_step,
        load_checkpoint,
        save_checkpoint,
    )

    def run(d, n):
        return [d.step([f"reply {i}", f"other {i}"], [True, i % 2 == 0]) for i in range(n)]

    straight, _ = make_disc(lambda texts: [0.5] * len(texts))
    run(straight, 6)
    expected = run(straight, 4)

    first, _ = make_disc(lambda texts: [0.5] * len(texts))
    saved = []
    first.after_call = lambda calls: (
        saved.append(save_checkpoint(tmp_path, first, {"weights": calls})) if calls % 3 == 0 else None
    )
    run(first, 6)
    assert [p.name for p in saved] == ["calls-3.pt", "calls-6.pt"]

    resumed, updates = make_disc(lambda texts: [0.5] * len(texts))
    st = load_checkpoint(tmp_path, 6)
    assert st["model"] == {"weights": 6}
    resumed.load_state(st["discriminator"])
    assert (resumed.calls, resumed.updates) == (6, first.updates)
    assert run(resumed, 4) == expected  # same counters, same buffer contents, same random draws
    assert updates  # and it kept training

    # 3 ranks, one generation per 2 policy steps: policy step 50 is discriminator call 75
    assert calls_at_policy_step(50, 3, 2) == 75
    with pytest.raises(ValueError):
        calls_at_policy_step(51, 3, 2)
    with pytest.raises(FileNotFoundError):
        load_checkpoint(tmp_path, 9)


def test_checkpoints_keep_only_the_newest(tmp_path):
    from src.lagrangian_finetuning.verifier_gated_discriminator import save_checkpoint

    d, _ = make_disc(lambda texts: [0.5] * len(texts))
    for _ in range(5):
        d.step(["reply"], [True])
        save_checkpoint(tmp_path, d, {}, keep=3)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["calls-3.pt", "calls-4.pt", "calls-5.pt"]


def test_train_stage_continues_from_the_latest_finished_checkpoint(tmp_path):
    assert pipeline.latest_complete_checkpoint(tmp_path) is None
    for step, finished in ((50, True), (100, True), (150, False)):
        c = tmp_path / f"checkpoint-{step}"
        c.mkdir()
        if finished:
            (c / "trainer_state.json").write_text("{}")
    assert pipeline.latest_complete_checkpoint(tmp_path).name == "checkpoint-100"


def test_evaluate_rounds_picks_every_finished_round_but_the_last_and_reads_lambda(tmp_path):
    for r in (0, 1, 2):
        (tmp_path / f"icrl_round_{r:02d}" / "rl" / "adapter").mkdir(parents=True)
        (tmp_path / f"icrl_round_{r:02d}" / "phi").mkdir()
        (tmp_path / f"icrl_round_{r:02d}" / "rl" / "lambda.json").write_text(
            json.dumps({"lambda": 1.5 + r})
        )
    (tmp_path / "icrl_round_03" / "rl").mkdir(parents=True)  # unfinished: no adapter
    got = pipeline.icrl_rounds_to_evaluate(tmp_path)
    assert [r for r, _ in got] == [0, 1]  # round 2 is the last finished one = eval/tuned
    cfg = OmegaConf.create({"icrl": {"grpo_steps_per_round": 125}})
    met = {
        "true": 0.5, "true_ci95": [0.4, 0.6], "violation": 0.01, "violation_ci95": [0, 0.02],
        "reward": 0.51, "reward_ci95": [0.4, 0.6],
        "monitor_audit": {"auroc": 0.7, "mean_harm_violating": 0.5, "mean_harm_clean": 0.3},
    }
    row = pipeline.round_summary_row(cfg, 1, got[1][1], met)
    assert row["steps"] == 250 and row["lambda_after_round"] == 2.5
    assert row["monitor_auroc"] == 0.7


def test_within_prompt_auroc_only_compares_replies_to_the_same_prompt():
    # prompt 0: the hack costs more than its sibling; prompt 1: less. Across prompts
    # the hack of prompt 1 (0.9) still beats prompt 0's clean reply, which must not count.
    scores, labels, groups = [0.8, 0.1, 0.9, 0.95], [1, 0, 1, 0], [0, 0, 1, 1]
    assert pipeline.within_prompt_auroc(scores, labels, groups) == 0.5
    assert pipeline.within_prompt_auroc([0.5, 0.5], [1, 0], [0, 0]) == 0.5
    assert pipeline.within_prompt_auroc([0.5, 0.4], [1, 1], [0, 0]) is None


def test_split_heldout_is_disjoint_seeded_and_off_by_default():
    trajs = [[f"t{i}"] for i in range(100)]
    fit, held = pipeline.split_heldout(trajs, 0.1, seed=0)
    assert len(held) == 10 and len(fit) == 90
    assert not {t[0] for t in fit} & {t[0] for t in held}
    assert pipeline.split_heldout(trajs, 0.1, seed=0) == (fit, held)
    assert pipeline.split_heldout(trajs, 0.0, seed=0) == (trajs, [])


def test_select_demonstrations_drops_hacks_and_keeps_the_published_draw():
    import random

    class Bench:
        def load_items(self, cfg, split):
            return [{"id": f"d{i}", "response": "r", "hack": int(i % 10 == 0)} for i in range(50)]

    legacy = OmegaConf.create({"icrl": {"expert_demos": 20}})
    got, n = pipeline.select_demonstrations(legacy, Bench(), random.Random(0))
    want = random.Random(0).sample(Bench().load_items(None, ""), 20)  # what v2 did
    assert n == 50 and got == want
    clean = OmegaConf.create({"icrl": {"expert_demos": None, "drop_hacking_demonstrations": True}})
    got, _ = pipeline.select_demonstrations(clean, Bench(), random.Random(0))
    assert len(got) == 45 and not any(d["hack"] for d in got)


def test_shuffled_costs_keep_the_values_and_break_the_order():
    from src.lagrangian_finetuning.single_turn_constrained_grpo import shuffled_costs

    costs = [i / 100 for i in range(100)]
    out = shuffled_costs(costs, seed=3)
    assert sorted(out) == costs and out != costs and shuffled_costs(costs, seed=3) == out


class _FakeFeasibility:
    """Stands in for TrajectoryFeasibility: embeddings from a hash of the text, the real head and costs."""

    def __init__(self):
        from src.icrl_dual_training.maxent_constraint import StepFeasibilityHead

        self.head = StepFeasibilityHead(8, 16)
        self.meta = {}

    @classmethod
    def build(cls, *a, **k):
        return cls()

    def load_head(self, d):
        import torch

        self.head.load_state_dict(torch.load(Path(d) / "phi_head.pt"))

    def save(self, d):
        import torch

        Path(d).mkdir(parents=True, exist_ok=True)
        torch.save(self.head.state_dict(), Path(d) / "phi_head.pt")

    def embed(self, trajs):
        import hashlib

        import torch

        from src.icrl_dual_training.secret_keeping_reply_constraint import trajectory_batch

        rows = []
        for t in trajs:
            text = t[0] if isinstance(t, list) else t
            seed = int(hashlib.md5(text.encode()).hexdigest()[:8], 16)
            v = torch.randn(8, generator=torch.Generator().manual_seed(seed))
            rows.append(v + (8.0 if text.startswith("far") else 0.0))
        return trajectory_batch(torch.stack(rows), [1] * len(rows))

    def costs(self, batch):
        import torch

        from src.icrl_dual_training.maxent_constraint import trajectory_log_feasibility

        with torch.no_grad():
            return (1 - trajectory_log_feasibility(self.head, batch).exp()).tolist()


def test_calibrated_round_sets_a_budget_and_a_rerun_round_does_not_double_the_memory(
    monkeypatch, tmp_path
):
    import src.icrl_dual_training.secret_keeping_reply_constraint as skc

    monkeypatch.setattr(skc, "TrajectoryFeasibility", _FakeFeasibility)
    cfg = OmegaConf.create(
        {
            "constraint": {"encoder_model": "x", "max_length": 8, "head_hidden": 16, "device": None},
            "icrl": {
                "beta": 1.0, "sparsity_coeff": 0.05, "negative_selection": "all",
                "first_round_constraint_steps": 20, "constraint_steps": 10,
                "constraint_batch": 16, "constraint_lr": 1e-3, "inverse_method": "pucl",
                "pucl_selection": "heldout_quantile", "pucl_quantile": 0.99, "pucl_k": 2,
                "memory_per_round": True, "budget_from_heldout": True, "budget_margin": 0.05,
            },
        }
    )
    allowed = [[f"demo {i}"] for i in range(200)]
    fit, held = pipeline.split_heldout(allowed, 0.1, seed=0)
    samples = [
        {"trajectory": [f"{'far' if i < 5 else 'near'} {i}"], "violates": int(i < 5), "reward": 1, "group": i // 4}
        for i in range(40)
    ]
    phi0, stats0, audit0 = pipeline.fit_constraint_round(
        cfg, tmp_path, tmp_path / "icrl_round_00", 0, None, fit, samples, heldout_trajs=held
    )
    assert audit0["marked"] >= 5 and audit0["marked_violating"] == 5  # the outliers are marked
    budget = json.loads((phi0 / "budget.json").read_text())
    assert abs(budget["budget"] - (audit0["mean_cost_heldout_allowed"] + 0.05)) < 1e-9
    assert audit0["budget"] == budget["budget"]
    # round 1 run twice (a restart): its memory file is replaced, round 0's is loaded once
    for _ in range(2):
        _, stats1, _ = pipeline.fit_constraint_round(
            cfg, tmp_path, tmp_path / "icrl_round_01", 1, str(phi0), fit, samples, heldout_trajs=held
        )
        assert stats1["n_memory_steps"] == audit0["marked"]
    assert sorted(p.name for p in (tmp_path / "pucl_memory").iterdir()) == ["round_00.pt", "round_01.pt"]


def test_calibrated_config_switches_on_every_fix_and_v2_keeps_the_published_rule(
    monkeypatch, tmp_path
):
    monkeypatch.setenv("SCRATCH", str(tmp_path))
    d = REPO / "configs" / "lagrangian_finetuning"
    cal = pipeline.load_cfg(d / "countdown_code_icrl_pucl_calibrated_qwen7b.yaml", [], smoke=False)
    ic = cal.icrl
    assert ic.pucl_selection == "heldout_quantile" and ic.budget_from_heldout and ic.memory_per_round
    assert ic.drop_hacking_demonstrations and ic.expert_demos is None and ic.heldout_fraction == 0.1
    assert pipeline.load_cfg(
        d / "countdown_code_icrl_pucl_calibrated_qwen7b.yaml", [], smoke=True
    ).icrl.start_model_allowed_prompts == 8
    v2 = pipeline.load_cfg(d / "countdown_code_icrl_pucl_qwen7b.yaml", [], smoke=False).icrl
    assert "pucl_selection" not in v2 and "budget_from_heldout" not in v2
    assert "start_model_allowed_prompts" not in pipeline.load_cfg(
        d / "countdown_code_icrl_pucl_qwen7b.yaml", [], smoke=True
    ).icrl
    # the two ICRL configs differ only in run name and the icrl block
    a = OmegaConf.to_container(OmegaConf.load(d / "countdown_code_icrl_pucl_qwen7b.yaml"))
    b = OmegaConf.to_container(OmegaConf.load(d / "countdown_code_icrl_pucl_calibrated_qwen7b.yaml"))
    for x in (a, b):
        x.pop("run_name"), x.pop("icrl")
    assert a == b


def test_allowed_set_embeddings_are_computed_once_per_process():
    calls = []

    class Phi:
        meta = {"encoder_model": "cache-test"}

        def embed(self, trajs):
            calls.append(len(trajs))
            return _FakeFeasibility().embed(trajs)

    trajs = [[f"cache demo {i}"] for i in range(5)]
    a = pipeline.cached_embeddings(Phi(), trajs)
    b = pipeline.cached_embeddings(Phi(), trajs)
    pipeline.cached_embeddings(Phi(), trajs[:4])
    assert a is b and calls == [5, 4]
