"""Unit tests for AdversarialDetector."""
import pytest
from icrl.core.types import DemoDataset, Trajectory, Transition
from icrl.trainer.adversarial_detector import AdversarialDetector, AdversarialDetectorConfig


def _make_traj(reward: float) -> Trajectory:
    t = Transition(obs=None, action=0, reward=reward, next_obs=None, done=True)
    return Trajectory(transitions=[t], total_reward=reward, total_cost=0.0)


@pytest.fixture
def demos():
    return DemoDataset(safe=[_make_traj(r) for r in [8.0, 9.0, 8.5, 9.0]])


def test_threshold_max(demos):
    d = AdversarialDetector(AdversarialDetectorConfig(mode="max"))
    threshold = d.fit(demos)
    assert threshold == pytest.approx(9.0)


def test_threshold_mean_plus_std(demos):
    d = AdversarialDetector(AdversarialDetectorConfig(mode="mean_plus_std", std_multiplier=1.0))
    threshold = d.fit(demos)
    import numpy as np
    rewards = [8.0, 9.0, 8.5, 9.0]
    expected = np.mean(rewards) + np.std(rewards)
    assert threshold == pytest.approx(expected)


def test_threshold_percentile(demos):
    d = AdversarialDetector(AdversarialDetectorConfig(mode="percentile", percentile=100.0))
    threshold = d.fit(demos)
    assert threshold == pytest.approx(9.0)


def test_is_unsafe_flags_above_threshold(demos):
    d = AdversarialDetector(AdversarialDetectorConfig(mode="max", min_reward_gap=0.01))
    d.fit(demos)
    assert not d.is_unsafe(_make_traj(9.0))    # exactly at threshold — safe
    assert not d.is_unsafe(_make_traj(9.005))  # within gap — safe
    assert d.is_unsafe(_make_traj(9.02))       # above threshold + gap — unsafe
    assert d.is_unsafe(_make_traj(10.0))       # clearly unsafe


def test_filter_unsafe_splits_correctly(demos):
    d = AdversarialDetector(AdversarialDetectorConfig(mode="max", min_reward_gap=0.01))
    d.fit(demos)
    trajs = [_make_traj(r) for r in [5.0, 9.0, 9.5, 10.0, 7.0]]
    safe, unsafe = d.filter_unsafe(trajs)
    assert len(safe) == 3   # 5.0, 9.0, 7.0
    assert len(unsafe) == 2  # 9.5, 10.0


def test_fit_not_called_raises():
    d = AdversarialDetector(AdversarialDetectorConfig())
    with pytest.raises(RuntimeError):
        _ = d.threshold


def test_stats(demos):
    d = AdversarialDetector(AdversarialDetectorConfig(mode="max"))
    d.fit(demos)
    stats = d.stats()
    assert stats["threshold"] == 9.0
    assert stats["n_safe_demos"] == 4
