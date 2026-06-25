"""Exact optimal-reward (theoretical-max eval return) per MiniGrid env.

MiniGrid gives reward = 1 - 0.9*(step_count/max_steps) on reaching the goal (0
on timeout). The theoretical-max *eval return* (the eval averages 10 random
layouts) is therefore 1 - 0.9*E[d*]/max_steps, where d* is the optimal action
count to reach the goal. We compute d* exactly with a BFS over
(x, y, dir, carrying, opened-doors) using the same action repertoire the agent
has (turn left/right, forward, pickup, toggle), and average the optimal reward
over many random layouts.

Run: PYTHONPATH=. python optimal_reward.py
The printed per-env means are the THEORETICAL_MAX values used in make_report_figs.py.
"""
from __future__ import annotations

from collections import deque

import numpy as np
import gymnasium as gym
import minigrid  # noqa: F401  (registers env ids)

ENVS = ["MiniGrid-DoorKey-5x5-v0", "MiniGrid-FourRooms-v0", "MiniGrid-MultiRoom-N6-v0"]
DIRVEC = [(1, 0), (0, 1), (-1, 0), (0, -1)]  # MiniGrid dir 0=right,1=down,2=left,3=up


def optimal_steps(env) -> int | None:
    """Minimum action count from the agent start to stepping onto the goal."""
    g = env.unwrapped.grid
    W, H = g.width, g.height
    ax, ay = env.unwrapped.agent_pos
    adir = int(env.unwrapped.agent_dir)

    goal = None
    for x in range(W):
        for y in range(H):
            c = g.get(x, y)
            if c is not None and c.type == "goal":
                goal = (x, y)
    if goal is None:
        return None

    def kind(x, y):
        if x < 0 or y < 0 or x >= W or y >= H:
            return "wall"
        c = g.get(x, y)
        return "empty" if c is None else c.type

    start = (ax, ay, adir, False, frozenset())
    q = deque([(start, 0)])
    seen = {start}
    while q:
        (x, y, d, carry, opened), steps = q.popleft()
        # turn left / right
        for nd in ((d - 1) % 4, (d + 1) % 4):
            ns = (x, y, nd, carry, opened)
            if ns not in seen:
                seen.add(ns); q.append((ns, steps + 1))
        dx, dy = DIRVEC[d]
        fx, fy = x + dx, y + dy
        ft = kind(fx, fy)
        fobj = g.get(fx, fy) if (0 <= fx < W and 0 <= fy < H) else None
        if ft == "goal":
            return steps + 1  # stepping onto the goal ends the episode
        passable = ft in ("empty", "floor")
        if ft == "door":
            passable = fobj.is_open or ((fx, fy) in opened)
        if ft == "key" and carry:
            passable = True  # the single key was picked up -> its old cell is empty
        if passable:
            ns = (fx, fy, d, carry, opened)
            if ns not in seen:
                seen.add(ns); q.append((ns, steps + 1))
        if ft == "key" and not carry:  # pickup
            ns = (x, y, d, True, opened)
            if ns not in seen:
                seen.add(ns); q.append((ns, steps + 1))
        if ft == "door" and not fobj.is_open and (fx, fy) not in opened:  # toggle open
            if (not fobj.is_locked) or carry:
                ns = (x, y, d, carry, opened | {(fx, fy)})
                if ns not in seen:
                    seen.add(ns); q.append((ns, steps + 1))
    return None


def theoretical_max(env_id: str, n: int = 300, base_seed: int = 10000) -> dict:
    env = gym.make(env_id)
    ms = env.unwrapped.max_steps
    rewards, steps = [], []
    for i in range(n):
        env.reset(seed=base_seed + i)
        d = optimal_steps(env)
        if d is None:
            continue
        steps.append(d); rewards.append(1 - 0.9 * d / ms)
    r = np.array(rewards)
    return {"env": env_id, "max_steps": ms, "n": len(r),
            "mean_steps": float(np.mean(steps)), "mean_reward": float(r.mean()),
            "std_reward": float(r.std())}


if __name__ == "__main__":
    for eid in ENVS:
        d = theoretical_max(eid)
        print(f"{d['env']:32s} max_steps={d['max_steps']:4d} n={d['n']} "
              f"opt_steps~{d['mean_steps']:5.1f}  theoretical_max={d['mean_reward']:.3f} "
              f"(+/-{d['std_reward']:.3f})")
