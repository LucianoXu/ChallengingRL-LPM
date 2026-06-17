from __future__ import annotations

import gymnasium as gym


class EpisodeRewardSplitWrapper(gym.Wrapper):
    """Accumulate per-episode extrinsic and intrinsic reward from the info dict
    (written by the RND/LPM intrinsic wrappers) and expose the sums at episode
    end as info['ep_extrinsic'] and info['ep_intrinsic'], so a Monitor created
    with info_keywords=('ep_extrinsic','ep_intrinsic') persists them to the
    monitor CSV.

    For baseline runs (no intrinsic wrapper below us) the intrinsic keys are
    absent, so ep_intrinsic accumulates 0 and ep_extrinsic accumulates the
    env's (pure-extrinsic) reward. Both keys are always emitted at episode end.
    """

    def __init__(self, env):
        super().__init__(env)
        self._ep_ext = 0.0
        self._ep_intr = 0.0

    def reset(self, **kwargs):
        self._ep_ext = 0.0
        self._ep_intr = 0.0
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._ep_ext += float(info.get("extrinsic_reward", reward))
        self._ep_intr += float(
            info.get("rnd_intrinsic_reward",
                     info.get("lpm_intrinsic_reward",
                              info.get("count_intrinsic_reward", 0.0))))
        if terminated or truncated:
            info["ep_extrinsic"] = self._ep_ext
            info["ep_intrinsic"] = self._ep_intr
        return obs, reward, terminated, truncated, info
