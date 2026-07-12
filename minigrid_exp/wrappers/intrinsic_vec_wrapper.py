from __future__ import annotations

import numpy as np
from stable_baselines3.common.vec_env import VecEnvWrapper


class IntrinsicVecWrapper(VecEnvWrapper):
    """Single shared intrinsic-reward model over all n_envs, updated once per
    PPO rollout (n_steps step_waits) — the paper's per-cycle update. Replaces the
    per-env, per-step gym.Wrapper for PPO so RND/LPM use ONE model on the
    aggregated rollout instead of n_envs independent per-step models.

    Reward per env = extrinsic + scaled intrinsic. The inner per-env
    EpisodeRewardSplitWrapper still emits ep_extrinsic (pure extrinsic, since no
    intrinsic wrapper sits below it) and a placeholder ep_intrinsic=0 at episode
    end; this wrapper overwrites ep_intrinsic with the per-env accumulated
    intrinsic so VecMonitor logs the true split.
    """

    def __init__(self, venv, model, n_steps: int):
        super().__init__(venv)
        self.model = model
        self.n_steps = int(n_steps)
        self._last_obs = None
        self._last_actions = None
        self._ep_intr = np.zeros(self.num_envs, dtype=np.float64)
        self._steps = 0

    def reset(self):
        obs = self.venv.reset()
        self._last_obs = np.asarray(obs, dtype=np.float32)
        self._ep_intr[:] = 0.0
        self._steps = 0
        return obs

    def step_async(self, actions):
        self._last_actions = np.asarray(actions)
        self.venv.step_async(actions)

    def step_wait(self):
        obs, rews, dones, infos = self.venv.step_wait()
        obs = np.asarray(obs, dtype=np.float32)
        # True next-obs is the terminal observation on done (SB3 auto-resets).
        next_obs = obs.copy()
        for i in range(self.num_envs):
            if dones[i]:
                next_obs[i] = np.asarray(infos[i]["terminal_observation"], dtype=np.float32)
        bonus = self.model.reward(self._last_obs, self._last_actions, next_obs)
        rews = np.asarray(rews, dtype=np.float32) + bonus
        self._ep_intr += bonus
        for i in range(self.num_envs):
            if dones[i]:
                infos[i]["ep_intrinsic"] = float(self._ep_intr[i])
                self._ep_intr[i] = 0.0
        self._last_obs = obs
        self._steps += 1
        if self._steps % self.n_steps == 0:
            self.model.update()
        return obs, rews, dones, infos

    def checkpoint_state(self):
        """Return the complete curiosity-wrapper state for crash-safe resume."""
        return {
            "version": 1,
            "n_steps": self.n_steps,
            "model": self.model.state_dict(),
            "last_obs": None if self._last_obs is None else self._last_obs.copy(),
            "last_actions": (
                None if self._last_actions is None else self._last_actions.copy()
            ),
            "episode_intrinsic": self._ep_intr.copy(),
            "steps": self._steps,
        }

    def load_checkpoint_state(self, state):
        if state.get("version") != 1:
            raise ValueError(
                f"Unsupported intrinsic-wrapper checkpoint version: {state.get('version')!r}"
            )
        if int(state["n_steps"]) != self.n_steps:
            raise ValueError(
                f"Checkpoint n_steps={state['n_steps']} does not match {self.n_steps}"
            )
        self.model.load_state_dict(state["model"])
        self._last_obs = (
            None
            if state["last_obs"] is None
            else np.array(state["last_obs"], dtype=np.float32, copy=True)
        )
        self._last_actions = (
            None
            if state["last_actions"] is None
            else np.array(state["last_actions"], copy=True)
        )
        self._ep_intr = np.array(
            state["episode_intrinsic"], dtype=np.float64, copy=True
        )
        self._steps = int(state["steps"])
