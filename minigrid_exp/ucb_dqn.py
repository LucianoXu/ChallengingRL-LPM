from __future__ import annotations

from collections import defaultdict

import numpy as np
import torch as th
from gymnasium import spaces
from stable_baselines3 import DQN


class UCBDQN(DQN):
    """
    DQN variant that uses state-action UCB for training-time action selection.

    Deterministic prediction stays greedy, so evaluation behaves like normal DQN.
    """

    def __init__(
        self,
        *args,
        ucb_coefficient: float = 1.0,
        ucb_state_round_decimals: int | None = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.ucb_coefficient = ucb_coefficient
        self.ucb_state_round_decimals = ucb_state_round_decimals
        self.ucb_action_counts = defaultdict(self._new_action_counts)

    def _new_action_counts(self):
        if not isinstance(self.action_space, spaces.Discrete):
            raise ValueError("UCBDQN only supports discrete action spaces.")
        return np.zeros(self.action_space.n, dtype=np.int64)

    def _excluded_save_params(self):
        return super()._excluded_save_params() + ["ucb_action_counts"]

    def _obs_to_key(self, obs):
        if isinstance(obs, dict):
            return tuple(
                (key, self._obs_to_key(value))
                for key, value in sorted(obs.items())
            )

        obs_array = np.asarray(obs)
        if (
            self.ucb_state_round_decimals is not None
            and np.issubdtype(obs_array.dtype, np.floating)
        ):
            obs_array = np.round(
                obs_array.astype(np.float32),
                decimals=self.ucb_state_round_decimals,
            )

        return (obs_array.shape, str(obs_array.dtype), obs_array.tobytes())

    @staticmethod
    def _split_observation_batch(observation, batch_size: int, vectorized: bool):
        if not vectorized:
            return [observation]

        if isinstance(observation, dict):
            return [
                {key: np.asarray(value)[index] for key, value in observation.items()}
                for index in range(batch_size)
            ]

        obs_array = np.asarray(observation)
        return [obs_array[index] for index in range(batch_size)]

    def _select_ucb_action(self, obs, q_values):
        counts = self.ucb_action_counts[self._obs_to_key(obs)]

        unvisited_actions = np.flatnonzero(counts == 0)
        if len(unvisited_actions) > 0:
            action = int(np.random.choice(unvisited_actions))
        else:
            total_visits = counts.sum()
            bonus = self.ucb_coefficient * np.sqrt(
                np.log(total_visits + 1.0) / counts
            )
            action = int(np.argmax(q_values + bonus))

        counts[action] += 1
        return action

    def predict(
        self,
        observation,
        state=None,
        episode_start=None,
        deterministic: bool = False,
    ):
        if deterministic:
            return super().predict(
                observation,
                state=state,
                episode_start=episode_start,
                deterministic=True,
            )

        obs_tensor, vectorized = self.policy.obs_to_tensor(observation)

        with th.no_grad():
            q_values = self.policy.q_net(obs_tensor).cpu().numpy()

        obs_batch = self._split_observation_batch(
            observation,
            batch_size=len(q_values),
            vectorized=vectorized,
        )
        actions = np.asarray(
            [
                self._select_ucb_action(obs, action_q_values)
                for obs, action_q_values in zip(obs_batch, q_values)
            ],
            dtype=np.int64,
        )

        if vectorized:
            return actions, state

        return actions[0], state
