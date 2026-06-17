from __future__ import annotations

import gymnasium as gym
import numpy as np
import torch as th
from gymnasium import spaces
from torch import nn


class RunningMeanStd:
    def __init__(self, shape=(), epsilon: float = 1e-4):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = epsilon

    def update(self, values):
        values = np.asarray(values, dtype=np.float64)
        batch_mean = np.mean(values, axis=0)
        batch_var = np.var(values, axis=0)
        batch_count = values.shape[0]
        self._update_from_moments(batch_mean, batch_var, batch_count)

    def _update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        total_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / total_count
        mean_square_a = self.var * self.count
        mean_square_b = batch_var * batch_count
        correction = np.square(delta) * self.count * batch_count / total_count
        new_var = (mean_square_a + mean_square_b + correction) / total_count

        self.mean = new_mean
        self.var = new_var
        self.count = total_count


class RNDNetwork(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, observations):
        return self.net(observations)


class RNDIntrinsicRewardWrapper(gym.Wrapper):
    """
    Adds Random Network Distillation intrinsic reward.

    The target network is fixed. The predictor is trained online to match it.
    Novel observations have larger prediction error and therefore larger bonus.
    """

    def __init__(
        self,
        env,
        reward_scale: float = 0.05,
        learning_rate: float = 1e-4,
        hidden_dim: int = 128,
        output_dim: int = 128,
        normalize_observations: bool = True,
        normalize_rewards: bool = True,
        observation_clip: float = 5.0,
        device: str = "auto",
        seed: int | None = None,
    ):
        super().__init__(env)
        self.reward_scale = reward_scale
        self.normalize_observations = normalize_observations
        self.normalize_rewards = normalize_rewards
        self.observation_clip = observation_clip
        self.device = self._resolve_device(device)

        self.input_dim = spaces.flatdim(self.observation_space)
        self.obs_rms = RunningMeanStd(shape=(self.input_dim,))
        self.reward_rms = RunningMeanStd(shape=())

        if seed is not None:
            th.manual_seed(seed)

        self.target = RNDNetwork(self.input_dim, hidden_dim, output_dim).to(self.device)
        self.predictor = RNDNetwork(self.input_dim, hidden_dim, output_dim).to(self.device)

        for parameter in self.target.parameters():
            parameter.requires_grad = False

        self.optimizer = th.optim.Adam(
            self.predictor.parameters(),
            lr=learning_rate,
        )
        self.loss_fn = nn.MSELoss(reduction="none")

    @staticmethod
    def _resolve_device(device: str):
        if device == "auto":
            return th.device("cuda" if th.cuda.is_available() else "cpu")
        return th.device(device)

    def _flatten_observation(self, obs):
        flat_obs = spaces.flatten(self.observation_space, obs)
        return flat_obs.astype(np.float32)

    def _normalize_observation(self, flat_obs):
        if not self.normalize_observations:
            return flat_obs

        normalized = (flat_obs - self.obs_rms.mean) / np.sqrt(self.obs_rms.var + 1e-8)
        return np.clip(normalized, -self.observation_clip, self.observation_clip).astype(
            np.float32
        )

    def _obs_to_tensor(self, obs):
        flat_obs = self._flatten_observation(obs)
        self.obs_rms.update(flat_obs[None, :])
        normalized_obs = self._normalize_observation(flat_obs)
        return th.as_tensor(
            normalized_obs[None, :],
            dtype=th.float32,
            device=self.device,
        )

    def _compute_and_train(self, obs):
        obs_tensor = self._obs_to_tensor(obs)

        with th.no_grad():
            target_features = self.target(obs_tensor)

        predictor_features = self.predictor(obs_tensor)
        per_feature_loss = self.loss_fn(predictor_features, target_features)
        per_sample_loss = per_feature_loss.mean(dim=1)
        raw_bonus = float(per_sample_loss.item())

        loss = per_sample_loss.mean()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.normalize_rewards:
            self.reward_rms.update(np.asarray([raw_bonus], dtype=np.float64))
            normalized_bonus = raw_bonus / np.sqrt(self.reward_rms.var + 1e-8)
        else:
            normalized_bonus = raw_bonus

        return raw_bonus, float(self.reward_scale * normalized_bonus)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        flat_obs = self._flatten_observation(obs)
        self.obs_rms.update(flat_obs[None, :])
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        raw_bonus, intrinsic_reward = self._compute_and_train(obs)
        extrinsic_reward = float(reward)
        total_reward = extrinsic_reward + intrinsic_reward

        info = dict(info)
        info["extrinsic_reward"] = extrinsic_reward
        info["rnd_raw_intrinsic_reward"] = raw_bonus
        info["rnd_intrinsic_reward"] = intrinsic_reward

        return obs, total_reward, terminated, truncated, info
