"""Tests for CountBasedExplorationWrapper — random-action rollouts only."""
import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from wrappers.env_factory import make_env


def _rollout(env, n_steps, seed=0):
    """Return list of (obs, reward, terminated, truncated, info) from random actions."""
    rng = np.random.default_rng(seed)
    obs, _ = env.reset(seed=seed)
    results = []
    for _ in range(n_steps):
        action = rng.integers(env.action_space.n)
        obs, reward, terminated, truncated, info = env.step(action)
        results.append((obs, reward, terminated, truncated, info))
        if terminated or truncated:
            obs, _ = env.reset()
    return results


def make_count_env(beta=1.0):
    return make_env(
        "MiniGrid-DoorKey-5x5-v0",
        intrinsic=True,
        noise=False,
        seed=0,
        training=True,
        method="count",
        beta=beta,
    )


def test_info_split_and_decomposition():
    """Every step must have extrinsic and count_intrinsic keys, and total == sum."""
    env = make_count_env(beta=1.0)
    results = _rollout(env, n_steps=50)
    env.close()
    for obs, reward, terminated, truncated, info in results:
        assert "extrinsic_reward" in info, "missing extrinsic_reward in info"
        assert "count_intrinsic_reward" in info, "missing count_intrinsic_reward in info"
        expected = info["extrinsic_reward"] + info["count_intrinsic_reward"]
        assert abs(reward - expected) < 1e-6, (
            f"reward decomposition mismatch: {reward} != {expected}"
        )


def test_first_visit_bonus_equals_reward_scale():
    """With beta=1.0 and reward_scale=1.0, the very first step visits a new obs
    so count_intrinsic_reward must equal exactly 1.0."""
    env = make_count_env(beta=1.0)
    rng = np.random.default_rng(42)
    env.reset(seed=42)
    action = rng.integers(env.action_space.n)
    obs, reward, terminated, truncated, info = env.step(action)
    env.close()
    assert info["count_intrinsic_reward"] == pytest.approx(1.0, abs=1e-8), (
        f"First-visit bonus should be 1.0 but got {info['count_intrinsic_reward']}"
    )


def test_bonus_decreases_with_revisits():
    """Over ~300 steps, states must recur, so min bonus < 1.0 and all in (0, 1]."""
    env = make_count_env(beta=1.0)
    results = _rollout(env, n_steps=300, seed=7)
    env.close()
    bonuses = [info["count_intrinsic_reward"] for _, _, _, _, info in results]
    assert all(0 < b <= 1.0 for b in bonuses), (
        f"Some bonus outside (0,1]: min={min(bonuses)}, max={max(bonuses)}"
    )
    assert min(bonuses) < 1.0, (
        f"Expected revisits (min bonus < 1.0), got min={min(bonuses)}"
    )
