from __future__ import annotations

import gymnasium as gym
import numpy as np
import torch as th
from gymnasium import spaces

from wrappers.lpm_wrapper import _MLP
from wrappers.rnd_wrapper import RunningMeanStd


class ICMIntrinsicRewardWrapper(gym.Wrapper):
    """Per-env Intrinsic Curiosity Module wrapper.

    PPO uses SharedICMModel above the VecEnv. This wrapper keeps the single-env
    path coherent for tests and dormant non-vector training paths.
    """

    def __init__(self, env, reward_scale: float = 0.005, learning_rate: float = 1e-3,
                 hidden_dim: int = 128, feature_dim: int = 128,
                 forward_loss_weight: float = 0.2,
                 normalize_observations: bool = True, normalize_rewards: bool = True,
                 observation_clip: float = 5.0, device: str = "auto", seed=None):
        super().__init__(env)
        if not isinstance(self.action_space, spaces.Discrete):
            raise ValueError("ICM wrapper supports discrete action spaces only.")
        self.reward_scale = float(reward_scale)
        self.forward_loss_weight = float(forward_loss_weight)
        self.normalize_observations = normalize_observations
        self.normalize_rewards = normalize_rewards
        self.observation_clip = observation_clip
        self.device = self._resolve_device(device)
        self.input_dim = spaces.flatdim(self.observation_space)
        self.num_actions = int(self.action_space.n)
        self.obs_rms = RunningMeanStd(shape=(self.input_dim,))
        self.reward_rms = RunningMeanStd(shape=())

        if seed is not None:
            th.manual_seed(seed)

        self.encoder = _MLP(self.input_dim, hidden_dim, feature_dim).to(self.device)
        self.forward_model = _MLP(feature_dim + self.num_actions, hidden_dim, feature_dim).to(self.device)
        self.inverse_model = _MLP(2 * feature_dim, hidden_dim, self.num_actions).to(self.device)
        params = (list(self.encoder.parameters()) + list(self.forward_model.parameters())
                  + list(self.inverse_model.parameters()))
        self.opt = th.optim.Adam(params, lr=learning_rate)
        self._prev_obs = None

    @staticmethod
    def _resolve_device(device):
        if device == "auto":
            return th.device("cuda" if th.cuda.is_available() else "cpu")
        return th.device(device)

    def _flatten(self, obs):
        return spaces.flatten(self.observation_space, obs).astype(np.float32)

    def _normalize(self, flat):
        if not self.normalize_observations:
            return flat
        z = (flat - self.obs_rms.mean) / np.sqrt(self.obs_rms.var + 1e-8)
        return np.clip(z, -self.observation_clip, self.observation_clip).astype(np.float32)

    def _action_one_hot(self, action):
        a = np.zeros((1, self.num_actions), dtype=np.float32)
        a[0, int(action)] = 1.0
        return th.as_tensor(a, dtype=th.float32, device=self.device)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        flat = self._flatten(obs)
        self.obs_rms.update(flat[None, :])
        self._prev_obs = self._normalize(flat)
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        flat = self._flatten(obs)
        self.obs_rms.update(flat[None, :])
        norm_next = self._normalize(flat)
        raw, intrinsic = self._compute_and_train(self._prev_obs, int(action), norm_next)
        self._prev_obs = norm_next

        info = dict(info)
        info["extrinsic_reward"] = float(reward)
        info["icm_raw_intrinsic_reward"] = raw
        info["icm_intrinsic_reward"] = intrinsic
        return obs, float(reward) + intrinsic, terminated, truncated, info

    def _compute_and_train(self, prev_norm, action, next_norm):
        st = th.as_tensor(prev_norm[None, :], dtype=th.float32, device=self.device)
        nt = th.as_tensor(next_norm[None, :], dtype=th.float32, device=self.device)
        at = th.as_tensor([action], dtype=th.long, device=self.device)
        ah = self._action_one_hot(action)

        phi_prev = self.encoder(st)
        phi_next = self.encoder(nt)
        pred_next = self.forward_model(th.cat([phi_prev, ah], dim=1))
        fwd_loss = ((pred_next - phi_next.detach()) ** 2).mean()
        inv_logits = self.inverse_model(th.cat([phi_prev, phi_next], dim=1))
        inv_loss = th.nn.functional.cross_entropy(inv_logits, at)
        loss = ((1.0 - self.forward_loss_weight) * inv_loss
                + self.forward_loss_weight * fwd_loss)
        self.opt.zero_grad(); loss.backward(); self.opt.step()

        raw = float(fwd_loss.item())
        if self.normalize_rewards:
            self.reward_rms.update(np.asarray([raw], dtype=np.float64))
            bonus = raw / np.sqrt(self.reward_rms.var + 1e-8)
        else:
            bonus = raw
        return raw, float(self.reward_scale * bonus)
