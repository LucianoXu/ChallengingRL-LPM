import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import gymnasium as gym
import minigrid  # noqa: F401  (registers MiniGrid env ids)
from minigrid.core.constants import OBJECT_TO_IDX, COLOR_TO_IDX, STATE_TO_IDX
from wrappers.noise_wrapper import ObservationNoiseWrapper


def test_zero_prob_is_noop():
    clean, _ = gym.make("MiniGrid-Empty-8x8-v0").reset(seed=0)
    wrapped = ObservationNoiseWrapper(
        gym.make("MiniGrid-Empty-8x8-v0"), noise_prob=0.0, seed=0)
    noisy, _ = wrapped.reset(seed=0)
    assert np.array_equal(clean["image"], noisy["image"])


def test_channels_stay_in_range():
    # noise_prob=1.0 corrupts every cell; assert no channel exceeds its valid max.
    env = ObservationNoiseWrapper(
        gym.make("MiniGrid-Empty-8x8-v0"), noise_prob=1.0, seed=1)
    env.reset(seed=1)
    omax, cmax, smax = (max(OBJECT_TO_IDX.values()),
                        max(COLOR_TO_IDX.values()),
                        max(STATE_TO_IDX.values()))
    for _ in range(50):
        obs, *_ = env.step(2)  # 2 = forward; deterministic given the seed
        img = obs["image"]
        assert img[..., 0].max() <= omax
        assert img[..., 1].max() <= cmax
        assert img[..., 2].max() <= smax


def test_mask_is_cell_level_and_fraction_matches():
    env = ObservationNoiseWrapper(
        gym.make("MiniGrid-Empty-8x8-v0"), noise_prob=0.3, seed=3)
    obs, _ = env.reset(seed=3)
    fracs = []
    for _ in range(200):
        obs, *_ = env.step(2)
        m = env.last_cell_mask
        assert m.shape == obs["image"].shape[:2]   # (H, W) — cell-level, not per-channel
        fracs.append(m.mean())
    assert abs(float(np.mean(fracs)) - 0.3) < 0.05


def test_same_seed_reproducible():
    def run():
        e = ObservationNoiseWrapper(
            gym.make("MiniGrid-Empty-8x8-v0"), noise_prob=0.5, seed=7)
        e.reset(seed=7)
        return [e.step(2)[0]["image"].copy() for _ in range(10)]
    a, b = run(), run()
    assert all(np.array_equal(x, y) for x, y in zip(a, b))
