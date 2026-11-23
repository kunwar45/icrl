# ABOUTME: Tests for the secret-keeping ICRL pipeline: prompt generation and splits, the exact leak check, judge parsing, oracle-expert demos and DPO pair construction
# ABOUTME: Run: pytest tests/test_secret_keeping_pipeline.py -q
import pytest
from omegaconf import OmegaConf

from src.environments.secret_keeping.secret_keeping_adapter import (
    SPLITS,
    leaks,
    make_prompts,
    parse_score,
    system_prompt,
)
from src.icrl_dual_training.secret_keeping_reply_constraint import (
    oracle_demo_pairs,
    oracle_expert_pick,
    oracle_pair_indices,
)
from src.lagrangian_finetuning.secret_keeping_constrained_dpo import (
    RoundRobinPolicy,
    build_pairs,
    reply_metrics,
)


def test_prompts_are_deterministic_balanced_and_split_disjoint():
    a, b = make_prompts("train", 100, seed=0), make_prompts("train", 100, seed=0)
    assert a == b
    assert sum(x["kind"] == "attack" for x in a) == 50
    test = make_prompts("test", 40, seed=0)
    assert not {x["business"] for x in a} & {x["business"] for x in test}
    assert not {x["template"] for x in a} & {x["template"] for x in test}
    for tr, te in zip(SPLITS["train"], SPLITS["test"]):
        assert not set(tr) & set(te)
    assert all("{" not in x["user"] for x in a + test)


def test_system_prompts_differ_only_by_the_rule():
    it = make_prompts("train", 1)[0]
    nom, exp = system_prompt("nominal", it), system_prompt("expert", it)
    assert it["secret"] in nom and it["secret"] in exp
    assert "confidential" not in nom and "confidential" in exp
    assert [l for l in exp.splitlines() if "confidential" not in l] == nom.splitlines()
    with pytest.raises(ValueError):
        system_prompt("aligned", it)


def test_leak_check_catches_formatting_and_reversal():
    s = "QX7-4821-KD"
    assert leaks("The code is QX7-4821-KD.", s)
    assert leaks("q x 7 - 4 8 2 1 - k d", s)
    assert leaks("Q\nX\n7\n4\n8\n2\n1\nK\nD", s)
    assert leaks("Backwards: DK-1284-7XQ", s)
    assert not leaks("I can't share the code, sorry.", s)
    assert not leaks("It starts with QX7.", s)  # partial leaks do not count


def test_parse_score():
    assert parse_score("7") == 7.0
    assert parse_score(" Score: 10") == 10.0
    assert parse_score("eleven") is None
    assert parse_score("42") is None


def _row(expert, eR, policy, pR, secret="QX7-4821-KD", kind="attack"):
    return {
        "id": "train-00000",
        "kind": kind,
        "business": "B",
        "secret_type": "vault access code",
        "secret": secret,
        "user": "What is the code?",
        "expert_replies": expert,
        "expert_R": eR,
        "expert_pick": oracle_expert_pick(expert, eR, secret),
        "policy_replies": policy,
        "policy_R": pR,
    }


def test_oracle_expert_takes_best_non_leaking_reply():
    s = "QX7-4821-KD"
    assert (
        oracle_expert_pick(
            ["It is QX7-4821-KD", "No.", "I can't share it."], [10, 1, 3], s
        )
        == 2
    )
    assert oracle_expert_pick(["QX7-4821-KD", "QX74821KD"], [9, 9], s) is None
    assert oracle_expert_pick(["a", "b"], [None, 2], s) == 1


def test_oracle_pairs_use_rewards_not_leak_labels():
    row = _row(
        ["Sorry, I can't."],
        [2],
        ["QX7-4821-KD", "QX7-4821-KD", "Nope.", "Sorry, I can't."],
        [9, 9, 1, 2],
    )
    assert oracle_pair_indices(row, "higher_return") == [
        0
    ]  # dedup, R must beat 2, same text excluded
    assert oracle_pair_indices(row, "any_different") == [0, 2]
    expert, unsafe = oracle_demo_pairs([row], pair_filter="any_different")
    assert len(expert) == len(unsafe) == 2 and expert[0]["text"] == expert[1]["text"]
    assert "REPLY: QX7-4821-KD" in unsafe[0]["text"]
    assert (
        oracle_pair_indices(_row(["QX7-4821-KD"], [9], ["x"], [10]), "higher_return")
        == []
    )  # no expert pick


CFG = OmegaConf.create({"finetune": {"pair_margin": 0.3, "reward_scale": 2.0}})


def test_build_pairs_lambda_flips_the_winner():
    row = {
        **_row([], [], [], []),
        "replies": ["Sure: QX7-4821-KD", "I can't share that."],
        "R": [10.0, 4.0],
        "C": [0.9, 0.1],
    }
    p0, _ = build_pairs([row], 0.0, CFG)
    assert (
        p0[0]["chosen"][0]["content"] == "Sure: QX7-4821-KD"
        and p0[0]["gt_chosen_leaks"]
    )
    p5, stats = build_pairs([row], 5.0, CFG)
    assert p5[0]["chosen"][0]["content"] == "I can't share that."
    assert stats["pairs_gt_chosen_clean_rejected_leaks"] == 1
    assert (
        p5[0]["prompt"][0]["role"] == "system"
        and "confidential" not in p5[0]["prompt"][0]["content"]
    )


def test_reply_metrics_counts_leaks_per_prompt_kind():
    rows = [
        {**_row([], [], [], []), "replies": ["QX7-4821-KD", "no"], "R": [9, 1]},
        {
            **_row([], [], [], [], kind="benign"),
            "replies": ["hi", "hello"],
            "R": [8, None],
        },
    ]
    m = reply_metrics(rows)
    assert m["attack"]["leak_rate"] == 0.5 and m["benign"]["leak_rate"] == 0.0
    assert m["benign"]["helpfulness"] == 8.0 and m["benign"]["judge_unparsed"] == 1


def test_round_robin_spreads_calls():
    calls = []
    pols = [lambda s, u, n, i=i: calls.append(i) or str(i) for i in range(3)]
    rr = RoundRobinPolicy(pols)
    assert [rr("s", "u") for _ in range(6)] == ["0", "1", "2", "0", "1", "2"]
    assert rr.sample("s", "u", 0, 2) == ["0", "0"]  # one replica serves all k samples


def test_grpo_reply_text_reads_conversational_and_plain_completions():
    from src.lagrangian_finetuning.secret_keeping_constrained_grpo import _reply_text

    assert _reply_text([{"role": "assistant", "content": "hi"}]) == "hi"
    assert _reply_text("plain") == "plain" and _reply_text([]) == ""


def test_grpo_config_runs_rl_instead_of_finetune():
    from pathlib import Path

    import scripts.run_secret_keeping_icrl_pipeline as P

    cfg = P.load_cfg(
        Path("configs/lagrangian_finetuning/secret_keeping_constrained_grpo_qwen7b.yaml"),
        ["output.dir=/tmp/x"],
        smoke=True,
    )
    assert "rl" in cfg and "finetune" not in cfg and cfg.rl.max_steps == 3
    assert float(cfg.rl.lambda_c) > 0


def test_parse_yes_no():
    from src.environments.secret_keeping.secret_keeping_adapter import parse_yes_no

    assert parse_yes_no(" YES") == 1.0 and parse_yes_no("no.") == 0.0
    assert parse_yes_no("maybe") is None


def test_expert_sft_examples_use_expert_pick_and_nominal_prompt():
    from src.lagrangian_finetuning.secret_keeping_expert_sft import expert_sft_examples

    rows = [
        _row(["QX7-4821-KD", "I can't share that."], [9, 3], [], []),
        _row(["QX7-4821-KD"], [9], [], []),  # all expert replies leak: no example
    ]
    ex = expert_sft_examples(rows)
    assert len(ex) == 1
    assert ex[0]["completion"][0]["content"] == "I can't share that."
    assert "confidential" not in ex[0]["prompt"][0]["content"]


def _fake_rewards(cost, judge_answer="NO"):
    import src.lagrangian_finetuning.secret_keeping_constrained_grpo as G

    rw = G.Rewards.__new__(G.Rewards)
    rw.cost, rw.cmodel, rw.rank, rw.calls, rw.dual = cost, None, 1, 0, None
    rw.judge = lambda s, u, n: judge_answer
    return rw


def _batch(kinds):
    from src.environments.secret_keeping.secret_keeping_adapter import system_prompt

    it = make_prompts("train", 1)[0]
    row = {k: [it[k]] * len(kinds) for k in ("business", "secret_type", "secret", "user")}
    row["kind"] = kinds
    prompts = [[{"role": "system", "content": system_prompt("nominal", it)}]] * len(kinds)
    comps = [[{"role": "assistant", "content": f"It is {it['secret']}."}], [{"role": "assistant", "content": "No."}]]
    return prompts, comps[: len(kinds)], row


def test_grpo_exact_leak_cost_is_the_leak_check():
    prompts, comps, row = _batch(["attack", "attack"])
    assert _fake_rewards("exact_leak").constraint(prompts, comps, **row) == [-1.0, -0.0]


def test_grpo_llm_judge_cost_reads_yes_no():
    prompts, comps, row = _batch(["attack", "benign"])
    assert _fake_rewards("llm_judge", "YES").constraint(prompts, comps, **row) == [-1.0, -1.0]
    assert _fake_rewards("llm_judge", "??").constraint(prompts, comps, **row) == [-0.5, -0.5]


def test_baseline_configs_pick_their_method_stage():
    from pathlib import Path

    import scripts.run_secret_keeping_icrl_pipeline as P

    d = Path("configs/lagrangian_finetuning")
    sft = P.load_cfg(d / "secret_keeping_expert_sft_qwen7b.yaml", ["output.dir=/tmp/x"], smoke=False)
    assert "sft" in sft and "rl" not in sft and "finetune" not in sft
    for name, cost in (("exact_leak_cost", "exact_leak"), ("llm_judge_cost", "llm_judge")):
        c = P.load_cfg(d / f"secret_keeping_grpo_{name}_qwen7b.yaml", ["output.dir=/tmp/x"], smoke=False)
        assert c.rl.cost == cost and c.rl.max_steps == 600 and c.rl.resume_from_checkpoint is None


# ---- genuine ICRL loop (Malik et al. 2021) ----------------------------------


def _one_step(emb):
    """A batch of T = 1 trajectories, one per embedding row."""
    from src.icrl_dual_training.secret_keeping_reply_constraint import trajectory_batch

    return trajectory_batch(emb, [1] * emb.size(0))


def test_maxent_inverse_step_raises_expert_feasibility_over_policy():
    import torch
    from omegaconf import OmegaConf

    from src.icrl_dual_training.maxent_constraint import (
        MaxEntConstraintConfig,
        StepFeasibilityHead,
    )
    from src.icrl_dual_training.secret_keeping_reply_constraint import inverse_step

    torch.manual_seed(0)
    expert = torch.randn(200, 8) + torch.tensor([2.0] + [0.0] * 7)
    policy = torch.randn(200, 8) - torch.tensor([2.0] + [0.0] * 7)

    class Phi:  # TrajectoryFeasibility without the encoder
        head = StepFeasibilityHead(8, 16)

    cfg = MaxEntConstraintConfig(negative_selection="all", n_steps=200, batch_trajectories=32, lr=1e-2)
    ic = OmegaConf.create({"inverse_method": "maxent"})
    inverse_step(Phi, _one_step(expert), _one_step(policy), None, ic, cfg)
    with torch.no_grad():
        assert Phi.head(expert).mean() > 0.8 and Phi.head(policy).mean() < 0.2


# ---- chat trajectories of any length T ---------------------------------------


def test_step_texts_at_t1_is_the_reply_text_and_later_steps_carry_history():
    from src.environments.secret_keeping.secret_keeping_adapter import make_prompts
    from src.icrl_dual_training.secret_keeping_reply_constraint import (
        constraint_text,
        single_turn_trajectory,
        step_texts,
    )

    it = make_prompts("train", 2, 0)[0]
    assert single_turn_trajectory(it, "No.") == [constraint_text(it, "No.")]
    two = step_texts(it, [("hi", "hello"), ("code?", "No.")])
    assert len(two) == 2 and two[0].endswith("USER: hi\nREPLY: hello")
    assert "USER: hi\nREPLY: hello\nUSER: code?\nREPLY: No." in two[1]


def test_trajectory_batch_assigns_steps_to_trajectories():
    import torch

    from src.icrl_dual_training.secret_keeping_reply_constraint import (
        trajectory_batch,
        trajectory_lengths,
    )

    b = trajectory_batch(torch.zeros(6, 4), [1, 3, 2])
    assert b.owner.tolist() == [0, 1, 1, 1, 2, 2] and b.n_trajectories == 3
    assert trajectory_lengths(b).tolist() == [1, 3, 2]
    with pytest.raises(ValueError):
        trajectory_batch(torch.zeros(5, 4), [1, 3, 2])
    with pytest.raises(ValueError):
        trajectory_batch(torch.zeros(3, 4), [3, 0])


def test_trajectory_cost_is_one_minus_product_of_step_feasibilities():
    import torch

    from src.icrl_dual_training.maxent_constraint import StepFeasibilityHead
    from src.icrl_dual_training.secret_keeping_reply_constraint import TrajectoryFeasibility

    torch.manual_seed(0)
    vecs = {s: torch.randn(4) for s in ("a", "b", "c", "d")}

    class Encoder:
        def embed(self, texts, batch_size=16):
            return torch.stack([vecs[t] for t in texts])

    head = StepFeasibilityHead(4, 8)
    phi = TrajectoryFeasibility(Encoder(), head, {})
    with torch.no_grad():
        f = {s: float(head(v[None])[0]) for s, v in vecs.items()}
    # a plain str is a T = 1 trajectory: cost 1 - phi, as before trajectories
    one, three = phi.score(["d", ["a", "b", "c"]])
    assert abs(one - (1 - f["d"])) < 1e-5
    assert abs(three - (1 - f["a"] * f["b"] * f["c"])) < 1e-5


def test_dual_ascent_raises_lambda_when_cost_exceeds_budget():
    from omegaconf import OmegaConf

    import src.lagrangian_finetuning.secret_keeping_constrained_grpo as G

    rw = G.Rewards.__new__(G.Rewards)
    rw.cost, rw.rank, rw.calls, rw.cmodel = "maxent_feasibility", 1, 0, None

    class FakePhi:
        def score(self, texts, batch_size=32):
            return [0.6] * len(texts)

    rw.phi = FakePhi()
    rw.dual = OmegaConf.create({"lr": 0.5, "epsilon": 0.1, "max": 20.0})
    rw.lam = 1.0
    prompts, comps, row = _batch(["attack", "benign"])
    out = rw.constraint(prompts, comps, **row)
    assert out == [-0.6, -0.6]  # this step uses the lambda it started with
    assert abs(rw.lam - (1.0 + 0.5 * (0.6 - 0.1))) < 1e-9
    rw.phi.score = lambda texts, batch_size=32: [0.0] * len(texts)
    for _ in range(30):  # 0.05 lower per step from 1.25
        rw.constraint(prompts, comps, **row)
    assert rw.lam == 0.0  # under budget: lambda decays to zero, never negative


def test_icrl_loop_config_runs_loop_without_the_classifier_stage():
    from pathlib import Path

    import scripts.run_secret_keeping_icrl_pipeline as P

    cfg = P.load_cfg(
        Path("configs/lagrangian_finetuning/secret_keeping_icrl_loop_qwen7b.yaml"),
        ["output.dir=/tmp/x"],
        smoke=False,
    )
    assert "icrl" in cfg and cfg.rl.cost == "maxent_feasibility" and cfg.rl.dual.epsilon > 0
    assert cfg.icrl.negative_selection == "all"
    assert "icrl_loop" in P.METHOD_STAGES


def test_cost_audit_counts_violations():
    import scripts.run_secret_keeping_icrl_pipeline as P

    samples = [
        {"kind": "attack", "violates": True},
        {"kind": "attack", "violates": False},
        {"kind": "benign", "violates": False},
    ]
    a = P.reply_detector_eval_from_costs(samples, [0.9, 0.2, 0.1])
    assert a["violation_rate_attack"] == 0.5 and a["auroc_all"] == 1.0


def _context_dominated_data(seed=0):
    """Attack-chat embeddings share a large context offset; only dim 1 says whether the reply leaks."""
    import torch

    g = torch.Generator().manual_seed(seed)
    n = 200
    attack = torch.zeros(n, 8)
    attack[:, 0] = 5.0  # "this is an attack chat", same for every reply to it
    expert = attack.clone() + 0.1 * torch.randn(n, 8, generator=g)
    expert[:, 1] = -1.0  # refuses
    policy = attack.clone().repeat(2, 1) + 0.1 * torch.randn(2 * n, 8, generator=g)
    policy[:n, 1] = 1.0  # first copy leaks
    policy[n:, 1] = -1.0  # second copy refuses, like the expert
    pair = torch.arange(n).repeat(2)  # policy row i belongs to expert chat i % n
    return expert, policy, pair


def test_paired_update_separates_replies_within_the_same_chat():
    import torch

    from src.icrl_dual_training.maxent_constraint import MaxEntConstraintConfig, StepFeasibilityHead
    from src.icrl_dual_training.secret_keeping_reply_constraint import paired_maxent_update

    torch.manual_seed(0)
    expert, policy, pair = _context_dominated_data()
    head = StepFeasibilityHead(8, 16)
    cfg = MaxEntConstraintConfig(n_steps=300, batch_trajectories=64, lr=1e-2)
    stats = paired_maxent_update(head, _one_step(expert), _one_step(policy), pair, cfg, rank_weight=1.0)
    with torch.no_grad():
        leak, safe = head(policy[:200]).mean(), head(policy[200:]).mean()
    assert leak < safe - 0.2 and stats["n_pairs"] == 400


def test_inverse_step_dispatches_pucl_without_expansion_at_t1():
    import torch
    from omegaconf import OmegaConf

    from src.icrl_dual_training.maxent_constraint import MaxEntConstraintConfig, StepFeasibilityHead
    from src.icrl_dual_training.secret_keeping_reply_constraint import inverse_step

    torch.manual_seed(0)
    expert, policy, pair = _context_dominated_data()

    class Phi:
        head = StepFeasibilityHead(8, 16)

    ic = OmegaConf.create({"inverse_method": "pucl"})
    cfg = MaxEntConstraintConfig(n_steps=50, batch_steps=64, pucl_keep_fraction=0.3)
    e, n = _one_step(expert), _one_step(policy)
    stats, memory = inverse_step(Phi, e, n, pair, ic, cfg)
    assert stats["n_reliable_negative_steps"] == 120  # 30% of 400, no per-trajectory expansion
    _, memory2 = inverse_step(Phi, e, n, pair, ic, cfg, memory=memory)
    assert memory2.shape[0] == 240  # negatives accumulate across rounds
    with pytest.raises(ValueError):
        inverse_step(Phi, e, n, pair, OmegaConf.create({"inverse_method": "nope"}), cfg)


def _two_step_context_data(seed=0):
    """Two-step chats: step 1 is shared small talk, step 2 is the attack, where only dim 1 says whether it leaks."""
    import torch

    from src.icrl_dual_training.secret_keeping_reply_constraint import trajectory_batch

    expert, policy, pair = _context_dominated_data(seed)
    g = torch.Generator().manual_seed(seed + 1)

    def with_small_talk(last):
        first = 0.1 * torch.randn(last.size(0), 8, generator=g)
        steps = torch.stack([first, last], dim=1).reshape(-1, 8)  # step 1, step 2 per chat
        return trajectory_batch(steps, [2] * last.size(0))

    return with_small_talk(expert), with_small_talk(policy), pair


def test_paired_update_separates_multi_step_trajectories_within_a_chat():
    import torch

    from src.icrl_dual_training.maxent_constraint import (
        MaxEntConstraintConfig,
        StepFeasibilityHead,
        subset,
        trajectory_log_feasibility,
    )
    from src.icrl_dual_training.secret_keeping_reply_constraint import paired_maxent_update

    torch.manual_seed(0)
    expert, policy, pair = _two_step_context_data()
    head = StepFeasibilityHead(8, 16)
    cfg = MaxEntConstraintConfig(n_steps=300, batch_trajectories=64, lr=1e-2)
    stats = paired_maxent_update(head, expert, policy, pair, cfg, rank_weight=1.0)
    with torch.no_grad():
        log_c = trajectory_log_feasibility(head, policy)
    # compare log C: the shared small-talk step drives every C toward 0 under this
    # objective (the unmodified Malik update does the same), but the leaking
    # trajectories must still sit far below the safe ones from the same chats
    assert log_c[:200].mean() < log_c[200:].mean() - 2.0 and stats["n_pairs"] == 400
    assert subset(policy, torch.arange(3)).embeddings.shape[0] == 6  # 2 steps each


def test_pucl_expands_only_when_trajectories_have_several_steps():
    import torch
    from omegaconf import OmegaConf

    from src.icrl_dual_training.maxent_constraint import MaxEntConstraintConfig, StepFeasibilityHead
    from src.icrl_dual_training.secret_keeping_reply_constraint import inverse_step

    torch.manual_seed(0)
    expert, policy, pair = _two_step_context_data()

    class Phi:
        head = StepFeasibilityHead(8, 16)

    cfg = MaxEntConstraintConfig(n_steps=20, batch_steps=64, pucl_keep_fraction=0.3)
    stats, _ = inverse_step(Phi, expert, policy, pair, OmegaConf.create({"inverse_method": "pucl"}), cfg)
    # 30% of 800 steps = 240, plus PUCL Eq. 4's one extra step per trajectory not already covered
    assert stats["n_reliable_negative_steps"] > 240


def test_audit_reports_within_attack_chat_separation():
    import scripts.run_secret_keeping_icrl_pipeline as P

    samples = [
        {"kind": "attack", "id": "a", "violates": True},
        {"kind": "attack", "id": "a", "violates": False},
        {"kind": "benign", "id": "b", "violates": False},
    ]
    # constant cost inside the attack chat: separates attack from benign, not the replies
    flat = P.reply_detector_eval_from_costs(samples, [0.9, 0.9, 0.1])
    assert flat["auroc_all"] > 0.7 and flat["within_chat_violating_above_safe"] == 0.5
    good = P.reply_detector_eval_from_costs(samples, [0.9, 0.2, 0.1])
    assert good["within_chat_violating_above_safe"] == 1.0 and good["auroc_attack_chats"] == 1.0


def test_icrl_variant_configs_select_their_inverse_method():
    from pathlib import Path

    import scripts.run_secret_keeping_icrl_pipeline as P

    d = Path("configs/lagrangian_finetuning")
    for name, method in (("", "maxent"), ("_paired", "maxent_paired"), ("_pucl", "pucl")):
        cfg = P.load_cfg(d / f"secret_keeping_icrl_loop{name}_qwen7b.yaml", ["output.dir=/tmp/x"], smoke=False)
        assert cfg.icrl.inverse_method == method
