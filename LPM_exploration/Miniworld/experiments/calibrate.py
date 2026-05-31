"""Phase-0 calibration: confirm headless render works + measure throughput.

Usage: python calibrate.py --variant noisy_tv --steps 300 --device cpu
"""
import argparse
import time

import numpy as np
import torch

import _paths
_paths.ensure_repo_on_path()
from miniworld_play.envs import VARIANT_TO_ID  # noqa: E402
import gymnasium as gym  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=list(VARIANT_TO_ID), default="noisy_tv")
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--obs-scale", type=float, default=1.0)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    w, h = int(160 * args.obs_scale), int(120 * args.obs_scale)
    env = gym.make(VARIANT_TO_ID[args.variant], obs_width=w, obs_height=h).unwrapped
    obs, info = env.reset(seed=0)
    assert obs.shape == (h, w, 3), obs.shape
    pos0 = [float(env.agent.pos[0]), float(env.agent.pos[2])]
    print(f"[ok] headless render: obs {obs.shape}, pos {pos0}")

    t0 = time.time()
    for i in range(args.steps):
        obs, r, term, trunc, info = env.step(env.action_space.sample())
        if trunc or term:
            obs, info = env.reset()
    dt = time.time() - t0
    print(f"env-only: {args.steps/dt:.1f} steps/s ({dt:.1f}s for {args.steps})")
    env.close()


if __name__ == "__main__":
    main()
