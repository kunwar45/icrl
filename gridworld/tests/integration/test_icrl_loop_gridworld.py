"""
Integration tests for the adversarial ICRL loop on the 4×4 GridWorld.

4×4 spatial constraint:
  Safe path  (col 1 detour): 5 steps → reward = 15.0
  Unsafe path (col 0 short): 3 steps → reward = 17.0
  Threshold = 15.0

Tests:
  1. Safe demos load with correct rewards and threshold
  2. AdversarialDetector flags unsafe policy, not safe policy
  3. Constraint loss decreases; feasibility gap increases after training
  4. ICRLTrainer runs without error; unsafe buffer grows; metrics file written
"""
from __future__ import annotations

import pytest
import torch

from icrl.core.types import DemoDataset
from icrl.demos.gridworld_demos import GridWorldDemoConfig, GridWorldDemoLoader
from icrl.embedders.mean_pool import MeanPoolEmbedder, MeanPoolEmbedderConfig
from icrl.envs.gridworld import GridWorldConfig, GridWorldEnv
from icrl.constraints.mlp_constraint import MLPConstraint, MLPConstraintConfig
from icrl.policies.rule_based import SafeRuleBasedPolicy, UnsafeRuleBasedPolicy
from icrl.trainer.adversarial_detector import AdversarialDetector, AdversarialDetectorConfig
from icrl.trainer.icrl_trainer import ICRLConfig, ICRLTrainer
from icrl.trainer.rollout_buffer import collect_rollouts


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def env():
    return GridWorldEnv(GridWorldConfig(grid_size=4, seed=0))


@pytest.fixture
def demos(env):
    return GridWorldDemoLoader(GridWorldDemoConfig(n_demos=10, seed=0, grid_size=4)).load()


@pytest.fixture
def embedder(demos, env):
    emb = MeanPoolEmbedder(MeanPoolEmbedderConfig(embed_dim=8, vocab_size=64))
    emb.build_vocab(demos.safe, env)
    return emb


@pytest.fixture
def constraint(embedder, env):
    return MLPConstraint(embedder, env, MLPConstraintConfig(hidden_dim=16, lr=1e-2))


@pytest.fixture
def detector():
    return AdversarialDetector(AdversarialDetectorConfig(mode="max", min_reward_gap=0.01))


# ── Safe demos ────────────────────────────────────────────────────────────────

class TestSafeDemos:
    def test_demo_count(self, demos):
        assert len(demos.safe) == 10

    def test_safe_demos_complete_task(self, demos):
        for traj in demos.safe:
            assert traj.total_reward > 0, f"Demo reward {traj.total_reward} should be positive"

    def test_safe_reward_is_15(self, demos, env):
        expected = env.safe_episode_reward()  # 20 - 5 = 15.0
        assert expected == pytest.approx(15.0)
        assert demos.safe_reward_threshold == pytest.approx(15.0, abs=0.1)

    def test_unsafe_reward_above_threshold(self, demos, env):
        # Unsafe shortcut: 3 steps → 17.0 > 15.0
        assert env.unsafe_episode_reward() == pytest.approx(17.0)
        assert env.unsafe_episode_reward() > demos.safe_reward_threshold

    def test_safe_path_avoids_col0(self, demos):
        """Safe demos should never visit (1,0) or (2,0)."""
        forbidden = {(1, 0), (2, 0)}
        for traj in demos.safe:
            for t in traj.transitions:
                r, c = int(t.obs[0]), int(t.obs[1])
                assert (r, c) not in forbidden, (
                    f"Safe demo visited constrained cell ({r},{c})"
                )


# ── Adversarial detector ──────────────────────────────────────────────────────

class TestAdversarialDetector:
    def test_threshold_set_after_fit(self, demos, detector):
        threshold = detector.fit(demos)
        assert threshold == pytest.approx(15.0, abs=0.1)

    def test_safe_policy_not_flagged(self, env, demos, detector):
        detector.fit(demos)
        safe_policy = SafeRuleBasedPolicy()
        rollouts = collect_rollouts(env, safe_policy, n_steps=100)
        _, unsafe = detector.filter_unsafe(rollouts)
        assert len(unsafe) == 0, f"Safe policy flagged {len(unsafe)} trajectories — should be 0"

    def test_unsafe_policy_flagged(self, env, demos, detector):
        detector.fit(demos)
        unsafe_policy = UnsafeRuleBasedPolicy()
        rollouts = collect_rollouts(env, unsafe_policy, n_steps=100)
        _, unsafe = detector.filter_unsafe(rollouts)
        assert len(unsafe) > 0, "Unsafe policy (col-0 shortcut) should be flagged"

    def test_unsafe_policy_reward_above_threshold(self, env, demos, detector):
        detector.fit(demos)
        unsafe_policy = UnsafeRuleBasedPolicy()
        rollouts = collect_rollouts(env, unsafe_policy, n_steps=30)
        complete = [t for t in rollouts if t.transitions and t.transitions[-1].done]
        for traj in complete:
            assert traj.total_reward == pytest.approx(17.0, abs=0.1)
            assert traj.total_reward > detector.threshold


# ── Constraint learning ───────────────────────────────────────────────────────

class TestConstraintLearning:
    def test_constraint_loss_decreases(self, demos, constraint, env):
        unsafe_policy = UnsafeRuleBasedPolicy()
        detector = AdversarialDetector(AdversarialDetectorConfig(mode="max"))
        detector.fit(demos)
        rollouts = collect_rollouts(env, unsafe_policy, n_steps=60)
        _, unsafe_trajs = detector.filter_unsafe(rollouts)

        if not unsafe_trajs:
            pytest.skip("No unsafe trajectories generated")

        first_loss = last_loss = None
        for _ in range(15):
            m = constraint.update(demos.safe[:4], unsafe_trajs[:4])
            if "constraint_loss" in m:
                if first_loss is None:
                    first_loss = m["constraint_loss"]
                last_loss = m["constraint_loss"]

        assert first_loss is not None
        assert last_loss < first_loss, f"Loss did not decrease: {first_loss:.4f} → {last_loss:.4f}"

    def test_feasibility_gap_increases(self, demos, constraint, env):
        """After training, safe feasibility should exceed unsafe feasibility."""
        unsafe_policy = UnsafeRuleBasedPolicy()
        detector = AdversarialDetector(AdversarialDetectorConfig(mode="max"))
        detector.fit(demos)
        rollouts = collect_rollouts(env, unsafe_policy, n_steps=60)
        _, unsafe_trajs = detector.filter_unsafe(rollouts)

        if not unsafe_trajs:
            pytest.skip("No unsafe trajectories")

        for _ in range(20):
            constraint.update(demos.safe[:4], unsafe_trajs[:4])

        with torch.no_grad():
            safe_feas   = float(constraint.feasibility(demos.safe[0]).item())
            unsafe_feas = float(constraint.feasibility(unsafe_trajs[0]).item())

        assert safe_feas > unsafe_feas, (
            f"safe feasibility {safe_feas:.3f} should > unsafe {unsafe_feas:.3f}"
        )


# ── ICRL trainer ──────────────────────────────────────────────────────────────

class TestICRLLoop:
    def test_loop_runs_without_error(self, env, demos, constraint, detector, tmp_path):
        cfg = ICRLConfig(
            n_iterations=5, n_rollout_steps=100, n_constraint_epochs=2,
            constraint_batch_size=4, min_unsafe_for_update=1,
            eval_every=2, checkpoint_every=100, log_dir=str(tmp_path / "run"),
        )
        policy = UnsafeRuleBasedPolicy()
        trainer = ICRLTrainer(env=env, policy=policy, constraint=constraint,
                              demos=demos, detector=detector, config=cfg)
        trainer.train()
        assert trainer._iteration == 4

    def test_unsafe_buffer_grows(self, env, demos, constraint, detector, tmp_path):
        cfg = ICRLConfig(
            n_iterations=3, n_rollout_steps=100, n_constraint_epochs=1,
            constraint_batch_size=4, min_unsafe_for_update=1,
            eval_every=10, checkpoint_every=100, log_dir=str(tmp_path / "buf"),
        )
        policy = UnsafeRuleBasedPolicy()
        trainer = ICRLTrainer(env=env, policy=policy, constraint=constraint,
                              demos=demos, detector=detector, config=cfg)
        trainer.train()
        assert len(trainer._unsafe_buffer) > 0

    def test_metrics_file_created(self, env, demos, constraint, detector, tmp_path):
        import os
        log_dir = str(tmp_path / "metrics")
        cfg = ICRLConfig(
            n_iterations=2, n_rollout_steps=50, n_constraint_epochs=1,
            constraint_batch_size=4, min_unsafe_for_update=1,
            eval_every=10, checkpoint_every=100, log_dir=log_dir,
        )
        policy = UnsafeRuleBasedPolicy()
        trainer = ICRLTrainer(env=env, policy=policy, constraint=constraint,
                              demos=demos, detector=detector, config=cfg)
        trainer.train()
        assert os.path.exists(os.path.join(log_dir, "metrics.jsonl"))
