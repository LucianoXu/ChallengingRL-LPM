from __future__ import annotations

from collections import defaultdict

import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CountBasedExplorationWrapper(gym.Wrapper):
    """Count-based (UCB-style) exploration bonus — the deep-RL analogue of
    UCB / optimism-under-uncertainty:

        intrinsic = reward_scale / sqrt(N(obs))

    N(obs) is the running visitation count of the (hashed) flattened
    observation. Mirrors the RND/LPM wrapper interface: adds the bonus to the
    extrinsic reward and logs the split into info. Counting the OBSERVATION
    (not a privileged pose) keeps it comparable to RND/LPM and makes it
    susceptible to observation noise (a deliberate contrast with LPM).
    """

    def __init__(self, env, reward_scale: float = 0.05, seed=None):
        super().__init__(env)
        self.reward_scale = reward_scale
        self.counts = defaultdict(int)

    def _key(self, obs):
        arr = spaces.flatten(self.observation_space, obs)
        return np.asarray(arr).astype(np.int64).tobytes()

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        key = self._key(obs)
        self.counts[key] += 1
        n = self.counts[key]
        raw = 1.0 / float(np.sqrt(n))
        intrinsic = float(self.reward_scale * raw)
        info = dict(info)
        info["extrinsic_reward"] = float(reward)
        info["count_raw_intrinsic_reward"] = raw
        info["count_intrinsic_reward"] = intrinsic
        return obs, float(reward) + intrinsic, terminated, truncated, info
