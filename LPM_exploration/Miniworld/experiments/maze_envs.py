"""Adapter exposing the three maze variants from miniworld_play as a factory.

Single source of truth for geometry is miniworld_play/envs.py — we do not
re-port it here.
"""
from __future__ import annotations

import _paths
_paths.ensure_repo_on_path()

import gymnasium as gym  # noqa: E402
from miniworld_play.envs import VARIANT_TO_ID  # noqa: E402

OBS_W, OBS_H = 160, 120


def make_env(variant: str, seed: int = 0, obs_scale: float = 1.0,
             max_episode_steps: int = 50000):
    """Return an unwrapped maze env for `variant` at the requested resolution."""
    if variant not in VARIANT_TO_ID:
        raise ValueError(f"unknown variant {variant!r}; choose {list(VARIANT_TO_ID)}")
    w, h = int(OBS_W * obs_scale), int(OBS_H * obs_scale)
    env = gym.make(
        VARIANT_TO_ID[variant], obs_width=w, obs_height=h,
        max_episode_steps=max_episode_steps,
    ).unwrapped
    env.reset(seed=seed)
    return env
