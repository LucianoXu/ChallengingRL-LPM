from __future__ import annotations

import numpy as np
import torch as th
import gymnasium as gym
from gymnasium import spaces
from torch import nn

from wrappers.rnd_wrapper import RunningMeanStd


class _MLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return self.net(x)


class LPMIntrinsicRewardWrapper(gym.Wrapper):
    """Learning Progress Monitoring (Hou et al. 2026), MLP / flat-obs port.

    Forward dynamics f_theta predicts the next (normalized) flat observation
    from (obs, action); epsilon = log(MSE) is its log prediction error (Eq 1).
    Error model g_phi predicts that log-error (Eq 2). Intrinsic reward
    r = g_phi - epsilon (Eq 3), gated to 0 until the error buffer is full
    (|D| = buffer_size, Alg 1 line 6). The reward is running-std normalized and
    scaled by reward_scale — the same reward-scale knob used in
    RNDIntrinsicRewardWrapper (applied post running-std normalization). However,
    the per-step effect differs: RND's raw bonus is a non-negative squared error
    (positive mean), while LPM's raw reward g_phi - log(MSE) is approximately
    mean-zero and signed. Beta should therefore be swept per method, never shared
    across methods. Faithful to
    LPM_exploration/Miniworld/experiments/models.py:LPMModel(reward_space="log").

    NOTE: PPO now uses the shared wrappers.intrinsic_vec_wrapper.IntrinsicVecWrapper
    (one model over all n_envs, updated once per rollout). This per-env wrapper
    serves only the dormant DQN path and the unit tests of the reward formula.
    """

    def __init__(self, env, reward_scale: float = 0.05, learning_rate: float = 1e-3,
                 hidden_dim: int = 128, buffer_size: int = 100,
                 normalize_observations: bool = True, normalize_rewards: bool = True,
                 observation_clip: float = 5.0, device: str = "auto", seed=None):
        super().__init__(env)
        self.reward_scale = reward_scale
        self.normalize_observations = normalize_observations
        self.normalize_rewards = normalize_rewards
        self.observation_clip = observation_clip
        self.device = self._resolve_device(device)

        if not isinstance(self.action_space, spaces.Discrete):
            raise ValueError("LPM wrapper supports discrete action spaces only.")
        self.num_actions = int(self.action_space.n)
        self.input_dim = spaces.flatdim(self.observation_space)

        self.obs_rms = RunningMeanStd(shape=(self.input_dim,))
        self.reward_rms = RunningMeanStd(shape=())

        if seed is not None:
            th.manual_seed(seed)

        self.forward_model = _MLP(self.input_dim + self.num_actions,
                                  hidden_dim, self.input_dim).to(self.device)
        self.error_model = _MLP(self.input_dim + self.num_actions,
                                hidden_dim, 1).to(self.device)
        self.fwd_opt = th.optim.Adam(self.forward_model.parameters(), lr=learning_rate)
        self.err_opt = th.optim.Adam(self.error_model.parameters(), lr=learning_rate)

        self.buffer_size = buffer_size
        self.buf = []          # list of (norm_obs: np.ndarray, action: int, mse: float)
        self._prev_obs = None  # normalized previous observation

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

    def _sa(self, norm_obs, action):
        a = np.zeros(self.num_actions, dtype=np.float32)
        a[int(action)] = 1.0
        return th.as_tensor(np.concatenate([norm_obs, a])[None, :],
                            dtype=th.float32, device=self.device)

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
        raw, intrinsic = self._compute_and_train(self._prev_obs, action, norm_next)
        self._prev_obs = norm_next

        info = dict(info)
        info["extrinsic_reward"] = float(reward)
        info["lpm_raw_intrinsic_reward"] = raw
        info["lpm_intrinsic_reward"] = intrinsic
        return obs, float(reward) + intrinsic, terminated, truncated, info

    def _compute_and_train(self, prev_norm, action, next_norm):
        sa = self._sa(prev_norm, action)
        target = th.as_tensor(next_norm[None, :], dtype=th.float32, device=self.device)

        with th.no_grad():
            mse = float(((self.forward_model(sa) - target) ** 2).mean().item())
            g = float(th.clamp(self.error_model(sa), -10.0, 10.0).item())

        self.buf.append((prev_norm, int(action), mse))
        if len(self.buf) > self.buffer_size:
            self.buf.pop(0)

        # Train forward dynamics online (one gradient step on this transition).
        fwd_loss = ((self.forward_model(sa) - target) ** 2).mean()
        self.fwd_opt.zero_grad(); fwd_loss.backward(); self.fwd_opt.step()

        # Train g_phi to regress log(MSE) on a minibatch from the buffer (Eq 2).
        n = min(32, len(self.buf))
        idx = np.random.choice(len(self.buf), n, replace=False)
        bs = np.stack([self.buf[i][0] for i in idx])
        ba = np.array([self.buf[i][1] for i in idx])
        be = np.array([self.buf[i][2] for i in idx], dtype=np.float32)
        ah = np.zeros((n, self.num_actions), dtype=np.float32)
        ah[np.arange(n), ba] = 1.0
        x = th.as_tensor(np.concatenate([bs, ah], axis=1), dtype=th.float32, device=self.device)
        logp = th.clamp(self.error_model(x), -10.0, 10.0)
        logt = th.log(th.as_tensor(be, device=self.device) + 1e-6).unsqueeze(1)
        err_loss = ((logp - logt) ** 2).mean()
        self.err_opt.zero_grad(); err_loss.backward(); self.err_opt.step()

        # Reward: gated to 0 until |D| = buffer_size (Alg 1 L6), then r = g - log(MSE).
        if len(self.buf) < self.buffer_size:
            return 0.0, 0.0
        raw = float(g - float(np.log(mse + 1e-6)))
        if self.normalize_rewards:
            self.reward_rms.update(np.asarray([raw], dtype=np.float64))
            bonus = raw / np.sqrt(self.reward_rms.var + 1e-8)
        else:
            bonus = raw
        return raw, float(self.reward_scale * bonus)
