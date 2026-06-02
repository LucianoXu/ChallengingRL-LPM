"""Train one A2C+intrinsic run on a maze variant; log per-update metrics + positions.

Example:
  python train_maze.py --method lpm --variant noisy_tv --seed 1 --steps 20000 \
      --csv-log results/lpm-noisy_tv-s1.csv --pos-log positions/lpm-noisy_tv-s1.npz
"""
from __future__ import annotations

import argparse
import csv
import os
import random
import time

import numpy as np
import torch

import _paths
_paths.ensure_repo_on_path()
import coverage as cov
import maze_envs
import models as M
from a2c import A2CNetwork, A2CAgent


def pick_device(name):
    if name != "auto":
        return name
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _agent_pos(env):
    return float(env.agent.pos[0]), float(env.agent.pos[2])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", choices=["lpm", "rnd", "icm", "mse", "none"], required=True)
    ap.add_argument("--variant", choices=["nonoise", "noisy_tv", "action_noise"], required=True)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--update-frequency", type=int, default=64)
    ap.add_argument("--lambda-intrinsic", type=float, default=1.0)   # paper C.2: λ=1
    ap.add_argument("--entropy-coef", type=float, default=0.03)      # paper C.2: 0.03
    ap.add_argument("--normalize-intrinsic", action="store_true",
                    help="running mean/std-normalize intrinsic reward (our deviation; "
                         "paper C.2 uses raw lambda*r, so this is OFF by default)")
    ap.add_argument("--obs-scale", type=float, default=1.0)
    ap.add_argument("--random-policy", action="store_true",
                    help="ignore the A2C policy and pick actions uniformly at random "
                         "(no learning at all); the pure random-walk control")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--csv-log", required=True)
    ap.add_argument("--pos-log", required=True)
    ap.add_argument("--log-interval", type=int, default=10)
    args = ap.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = pick_device(args.device)

    env = maze_envs.make_env(args.variant, seed=args.seed, obs_scale=args.obs_scale)
    obs_shape = env.observation_space.shape          # (H,W,C)
    input_shape = (obs_shape[2], obs_shape[0], obs_shape[1])
    num_actions = env.action_space.n

    model = M.build_model(args.method, input_shape, num_actions, device)
    policy = A2CNetwork(input_shape, num_actions).to(device)
    agent = A2CAgent(policy, num_actions, device=device,
                     lambda_intrinsic=args.lambda_intrinsic,
                     entropy_coef=args.entropy_coef,
                     normalize_intrinsic=args.normalize_intrinsic)

    os.makedirs(os.path.dirname(args.csv_log) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(args.pos_log) or ".", exist_ok=True)
    cols = ["update", "step", "frames", "fps", "visited_count", "coverage_frac",
            "beyond_wall_frac", "time_at_wall_frac", "int_rew_mean",
            "pred_loss", "unc_loss", "fwd_loss", "rnd_loss",
            "policy_loss", "value_loss", "entropy"]
    cf = open(args.csv_log, "w", newline="")
    writer = csv.DictWriter(cf, fieldnames=cols); writer.writeheader()

    xs, zs, acts, sticky = [], [], [], []
    state, info = env.reset(seed=args.seed)
    x0, z0 = _agent_pos(env)
    xs.append(x0); zs.append(z0); acts.append(-1); sticky.append(False)

    update_i = 0; t0 = time.time(); int_accum = []
    for step in range(1, args.steps + 1):
        if args.random_policy:
            a, lp, v = random.randrange(num_actions), 0.0, 0.0
        else:
            a, lp, v = agent.select_action(state)
        ns, r, term, trunc, info = env.step(a)
        done = term or trunc
        ir = 0.0
        if not args.random_policy:
            ir = model.reward(state, ns, a)
            agent.memory.add(state, a, r, ir, ns, done, lp, v)
        int_accum.append(ir)
        xs.append(info["pos"][0]); zs.append(info["pos"][1])
        acts.append(int(info.get("action_id", a)))
        sticky.append(bool(info.get("sticky_replayed", False)))
        state = ns

        if step % args.update_frequency == 0:
            if args.random_policy:
                mloss, aloss = {}, {}
            else:
                mloss = model.update(
                    torch.FloatTensor(np.array(agent.memory.states)).to(device),
                    torch.FloatTensor(np.array(agent.memory.next_states)).to(device),
                    torch.LongTensor(agent.memory.actions).to(device))
                aloss = agent.update()
            update_i += 1
            if update_i % args.log_interval == 0:
                cm = cov.coverage_metrics(np.array(xs), np.array(zs))
                row = {c: "" for c in cols}
                row.update({"update": update_i, "step": step, "frames": step,
                            "fps": round(step / (time.time() - t0), 1),
                            "int_rew_mean": round(float(np.mean(int_accum)), 5)})
                row.update({k: round(v2, 5) for k, v2 in cm.items()})
                for k, v2 in {**mloss, **aloss}.items():
                    if k in row:
                        row[k] = round(float(v2), 5)
                writer.writerow(row); cf.flush()
                int_accum = []
        if done:
            state, info = env.reset()

    cf.close()
    np.savez_compressed(args.pos_log, step=np.arange(len(xs), dtype=np.int32),
                        x=np.array(xs, dtype=np.float32), z=np.array(zs, dtype=np.float32),
                        action=np.array(acts, dtype=np.int16), sticky=np.array(sticky))
    print(f"[done] {args.method}-{args.variant}-s{args.seed}: "
          f"coverage={cov.coverage_metrics(np.array(xs), np.array(zs))}")
    env.close()


if __name__ == "__main__":
    main()
