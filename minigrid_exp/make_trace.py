"""Render a trained policy's trajectory as a GIF.

Handles the `<env>__<variant>__<method>__seed_<n>[__tag]` run-name scheme.
The chunked checkpoint is `MODELS_DIR/<run_name>.zip`.

Usage:
  PYTHONPATH=. python make_trace.py <run_name> [<run_name> ...]
"""
from __future__ import annotations

import os
import re
import sys

import imageio.v2 as imageio

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from algorithms import get_algorithm_class  # noqa: E402
from wrappers.env_factory import make_env  # noqa: E402
from config import MODELS_DIR, EXPR_DATA  # noqa: E402

RUN_RE = re.compile(
    r"(?P<env>.+?)__(?P<variant>baseline_no_noise|baseline_noise|"
    r"intrinsic_no_noise|intrinsic_noise)__(?P<method>rnd|lpm|icm|none|entropy|count)__seed_(?P<seed>\d+)")


def render(run_name, max_episodes=40, max_steps=300, fps=8, prefer_solved=True):
    m = RUN_RE.match(run_name)
    if not m:
        raise ValueError(f"cannot parse run_name: {run_name}")
    env_id, variant, method, seed = (m.group("env"), m.group("variant"),
                                     m.group("method"), int(m.group("seed")))
    noise = variant.endswith("_noise") and not variant.endswith("no_noise")

    # Eval-style env (no intrinsic wrapper), render to RGB.
    env = make_env(env_id, intrinsic=False, noise=noise, seed=seed, training=False, method=method)
    model = get_algorithm_class().load(str(MODELS_DIR / f"{run_name}.zip"))

    # Search episodes; keep the best (highest-reward) one. If prefer_solved, stop
    # at the first solved episode (reward>0) so the GIF shows a success.
    best_frames, best_r, n_solved = None, -1.0, 0
    for ep in range(max_episodes):
        obs, _ = env.reset(seed=seed + 1000 + ep)
        fr = [env.render()]
        done = False; t = 0; epr = 0.0
        while not done and t < max_steps:
            action, _ = model.predict(obs, deterministic=True)
            obs, r, term, trunc, _ = env.step(action)
            done = term or trunc; t += 1; epr += float(r)
            fr.append(env.render())
        n_solved += int(epr > 0)
        if epr > best_r:
            best_frames, best_r = fr, epr
        if prefer_solved and epr > 0:
            break
    env.close()

    out_dir = os.path.join(str(EXPR_DATA), "figures", "traces")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, run_name.replace("/", "_") + ".gif")
    imageio.mimsave(path, best_frames, fps=fps)
    print(f"saved {path}: chosen-episode reward={best_r:.3f}, {len(best_frames)} frames "
          f"({n_solved} solved in the search)")


if __name__ == "__main__":
    for rn in sys.argv[1:]:
        render(rn)
