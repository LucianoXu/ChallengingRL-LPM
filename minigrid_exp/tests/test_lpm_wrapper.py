import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from wrappers.env_factory import make_env
from wrappers.lpm_wrapper import LPMIntrinsicRewardWrapper


def _rollout(env, n, seed=0):
    env.reset(seed=seed)
    rows = []
    for _ in range(n):
        obs, r, term, trunc, info = env.step(env.action_space.sample())
        rows.append(info)
        if term or trunc:
            env.reset()
    return rows


def test_info_has_extrinsic_intrinsic_split():
    env = make_env("MiniGrid-Empty-8x8-v0", intrinsic=True, noise=False,
                   seed=0, training=True, method="lpm", beta=0.05)
    inner = env
    while not isinstance(inner, LPMIntrinsicRewardWrapper):
        inner = inner.env
    rows = _rollout(env, 5)
    assert all("lpm_intrinsic_reward" in r and "extrinsic_reward" in r for r in rows)


def test_reward_gated_to_zero_before_buffer_fills():
    # buffer_size=5: the first 4 steps must be exactly 0.0 (no early activation),
    # and at least one step from step 5 onward must be non-zero (activation occurs).
    buffer_size = 5
    base = make_env("MiniGrid-Empty-8x8-v0", intrinsic=False, noise=False,
                    seed=0, training=True)
    env = LPMIntrinsicRewardWrapper(base, buffer_size=buffer_size, reward_scale=1.0, seed=0)
    rows = _rollout(env, 12)
    pre = [r["lpm_intrinsic_reward"] for r in rows[:buffer_size - 1]]
    post = [r["lpm_intrinsic_reward"] for r in rows[buffer_size - 1:]]
    assert all(x == 0.0 for x in pre), \
        f"Expected zeros before activation, got: {pre}"
    assert any(x != 0.0 for x in post), \
        f"Expected non-zero after activation, got: {post}"


def test_reward_nonzero_after_buffer_fills():
    base = make_env("MiniGrid-Empty-8x8-v0", intrinsic=False, noise=False,
                    seed=1, training=True)
    env = LPMIntrinsicRewardWrapper(base, buffer_size=5, reward_scale=1.0, seed=1)
    rows = _rollout(env, 60)
    later = [r["lpm_intrinsic_reward"] for r in rows[20:]]
    assert any(abs(x) > 0.0 for x in later), later
