# 3D-Maze Exploration Comparison + Coverage-Heatmap Evolution — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a CLI-driven A2C training pipeline for the LPM Miniworld 3D maze that compares intrinsic-motivation exploration methods (LPM / RND / ICM / MSE / none) across the three noise variants, logs per-step agent positions, and produces coverage curves + a new coverage-heatmap-evolution figure.

**Architecture:** Extract the maze notebooks' A2C + intrinsic models into importable modules under `LPM_exploration/Miniworld/experiments/`; reuse the env geometry from `miniworld_play/envs.py`; mirror the Ms Pac-Man `run_grid.py` / `analyze.py` infrastructure. Single environment per run (faithful to notebooks), CPU/MPS.

**Tech Stack:** Python 3.11, torch, numpy, matplotlib, gymnasium 1.2.3, miniworld 2.1.0, in `LPM_exploration/.venv`.

**Spec:** `docs/superpowers/specs/2026-05-31-maze-exploration-comparison-design.md`

**Reference source (verbatim notebook export):** `/tmp/maze_nb_with_noisyTV.txt` (A2C at lines 298-527 / 1126-1363; models at 164-296 / 879-1124). The canonical extraction target.

---

## Conventions

- Venv python: `LPM_exploration/.venv/bin/python` (call as `$PY` below).
- All new code lives under `LPM_exploration/Miniworld/experiments/` (the "pkg").
- Run tests with `PYTORCH_ENABLE_MPS_FALLBACK=1 $PY -m pytest <path> -v`.
- Obs from the env are `(120, 160, 3)` HWC uint8. Models' `input_shape` is the
  channel-first triple `(3, 120, 160)` (matches the notebooks). The shared
  `CNNFeatureExtractor.forward` does `x/255` then `permute(0,3,1,2)`, so tensors
  are fed as `(N, H, W, C)` float.
- Action space: `Discrete(5)` (0=turn_left, 1=turn_right, 2=move_forward,
  3=move_back, 4=variant-special). `num_actions = 5`.

---

## Phase 0 — Scaffolding + calibration

### Task 0.1: Package skeleton + import path

**Files:**
- Create: `LPM_exploration/Miniworld/experiments/__init__.py` (empty)
- Create: `LPM_exploration/Miniworld/experiments/_paths.py`
- Create: `LPM_exploration/Miniworld/experiments/tests/__init__.py` (empty)

- [ ] **Step 1: Write `_paths.py`** — resolves the repo root and puts it on
  `sys.path` so the pkg can `import miniworld_play.envs`.

```python
"""Path helpers: make `miniworld_play` importable from the experiments pkg."""
import os
import sys

# experiments/ -> Miniworld/ -> LPM_exploration/ -> <repo root>
_THIS = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(_THIS, "..", "..", ".."))
PKG_DIR = _THIS


def ensure_repo_on_path() -> str:
    if REPO_ROOT not in sys.path:
        sys.path.insert(0, REPO_ROOT)
    return REPO_ROOT
```

- [ ] **Step 2: Verify import works**

Run: `cd LPM_exploration/Miniworld/experiments && PYTHONPATH=../../.. ../../../.venv/bin/python -c "import _paths; _paths.ensure_repo_on_path(); import miniworld_play.envs as e; print(e.VARIANT_TO_ID)"`
Expected: prints the dict `{'nonoise': ..., 'noisy_tv': ..., 'action_noise': ...}`.

- [ ] **Step 3: Commit**

```bash
git add LPM_exploration/Miniworld/experiments/
git commit -m "maze-exp: package skeleton + repo-path helper"
```

### Task 0.2: Headless render + throughput calibration script

**Files:**
- Create: `LPM_exploration/Miniworld/experiments/calibrate.py`

- [ ] **Step 1: Write `calibrate.py`** — builds each variant headless, times
  N random steps incl. a dummy decoder forward, reports steps/sec on the chosen
  device. This sets the per-run `--steps` budget; no test (it is a measurement).

```python
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
    print(f"[ok] headless render: obs {obs.shape}, pos {info['pos']}")

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
```

- [ ] **Step 2: Run env-only calibration on CPU**

Run: `cd LPM_exploration/Miniworld/experiments && ../../../.venv/bin/python calibrate.py --variant noisy_tv --steps 300`
Expected: prints `[ok] headless render: obs (120, 160, 3) ...` and a steps/s figure. **Record the number** — it bounds the budget.

- [ ] **Step 3: Commit**

```bash
git add LPM_exploration/Miniworld/experiments/calibrate.py
git commit -m "maze-exp: headless render + throughput calibration script"
```

---

## Phase 1 — Geometry/coverage helpers + env adapter

### Task 1.1: `coverage.py` (pure, unit-tested)

**Files:**
- Create: `LPM_exploration/Miniworld/experiments/coverage.py`
- Test: `LPM_exploration/Miniworld/experiments/tests/test_coverage.py`

Maze geometry (world units): room1 x[0,4] z[0,8]; room2 x[14,18] z[0,8];
room3 (noise wall) x[0,18] z[8,8.1]; room4 x[0,18] z[8.1,12]. Grid is
`NX=72 × NZ=48` cells of size 0.25; `ix=int(x*4)`, `iz=int(z*4)`.

- [ ] **Step 1: Write the failing test**

```python
import numpy as np
import coverage as cov


def test_to_cell_basic():
    assert cov.to_cell(0.0, 0.0) == (0, 0)
    assert cov.to_cell(2.0, 1.0) == (8, 4)         # agent spawn
    assert cov.to_cell(17.99, 11.99) == (71, 47)   # far corner, clamped in-range


def test_reachable_mask_excludes_gap_and_counts_rooms():
    m = cov.reachable_mask()
    assert m.shape == (72, 48)
    # The gap between rooms 1 and 2 (x in [4,14], z in [0,8]) is unreachable.
    assert not m[cov.to_cell(9.0, 4.0)]
    # Inside room1, room2, room4 are reachable.
    assert m[cov.to_cell(2.0, 4.0)]
    assert m[cov.to_cell(16.0, 4.0)]
    assert m[cov.to_cell(9.0, 10.0)]
    # Reachable count is stable (regression guard).
    assert int(m.sum()) == cov.reachable_count()


def test_beyond_wall_mask_is_room4_only():
    bw = cov.beyond_wall_mask()
    assert bw[cov.to_cell(9.0, 10.0)]      # room4
    assert not bw[cov.to_cell(2.0, 4.0)]   # room1 (below wall)
    assert (bw & ~cov.reachable_mask()).sum() == 0  # subset of reachable


def test_coverage_metrics_from_positions():
    xs = np.array([2.0, 2.0, 9.0, 9.0])   # two cells in room1, two in room4
    zs = np.array([4.0, 4.0, 10.0, 10.5])
    cm = cov.coverage_metrics(xs, zs)
    assert cm["visited_count"] == 3        # (8,16) once + (36,40)+(36,42)
    assert 0 < cm["coverage_frac"] <= 1
    assert cm["beyond_wall_frac"] > 0
    assert 0 <= cm["time_at_wall_frac"] <= 1
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd LPM_exploration/Miniworld/experiments && PYTHONPATH=. ../../../.venv/bin/python -m pytest tests/test_coverage.py -v`
Expected: FAIL (module `coverage` not found / functions missing).

- [ ] **Step 3: Implement `coverage.py`**

```python
"""Maze-geometry coverage helpers. Pure numpy; no torch, no env."""
from __future__ import annotations

import numpy as np

NX, NZ = 72, 48           # grid cells (18*4, 12*4)
CELL = 0.25               # world units per cell
# (min_x, max_x, min_z, max_z) per room; room3 is the thin noise wall.
ROOMS = [
    (0.0, 4.0, 0.0, 8.0),     # room1
    (14.0, 18.0, 0.0, 8.0),   # room2
    (0.0, 18.0, 8.0, 8.1),    # room3 (noise wall)
    (0.0, 18.0, 8.1, 12.0),   # room4
]
WALL_Z = 8.0
WALL_BAND = (7.5, 8.5)    # "fascination with noise" band


def to_cell(x: float, z: float) -> tuple[int, int]:
    ix = min(NX - 1, max(0, int(x * 4)))
    iz = min(NZ - 1, max(0, int(z * 4)))
    return ix, iz


def _cell_centers():
    xs = (np.arange(NX) + 0.5) * CELL
    zs = (np.arange(NZ) + 0.5) * CELL
    return xs, zs


def reachable_mask() -> np.ndarray:
    """Cells whose centre lies inside any room rectangle."""
    xs, zs = _cell_centers()
    X, Z = np.meshgrid(xs, zs, indexing="ij")   # (NX, NZ)
    m = np.zeros((NX, NZ), dtype=bool)
    for x0, x1, z0, z1 in ROOMS:
        m |= (X >= x0) & (X < x1) & (Z >= z0) & (Z < z1)
    return m


def beyond_wall_mask() -> np.ndarray:
    """Reachable cells past the noise wall (room4, z >= 8.1)."""
    xs, zs = _cell_centers()
    X, Z = np.meshgrid(xs, zs, indexing="ij")
    return reachable_mask() & (Z >= 8.1)


def reachable_count() -> int:
    return int(reachable_mask().sum())


def occupancy_grid(xs: np.ndarray, zs: np.ndarray) -> np.ndarray:
    """Count of visits per cell from position arrays."""
    g = np.zeros((NX, NZ), dtype=np.int64)
    ix = np.clip((np.asarray(xs) * 4).astype(int), 0, NX - 1)
    iz = np.clip((np.asarray(zs) * 4).astype(int), 0, NZ - 1)
    np.add.at(g, (ix, iz), 1)
    return g


def coverage_metrics(xs: np.ndarray, zs: np.ndarray) -> dict:
    g = occupancy_grid(xs, zs)
    visited = g > 0
    reach = reachable_mask()
    bw = beyond_wall_mask()
    reach_n = int(reach.sum())
    bw_n = int(bw.sum())
    zs = np.asarray(zs)
    in_band = (zs >= WALL_BAND[0]) & (zs <= WALL_BAND[1])
    return {
        "visited_count": int(visited.sum()),
        "coverage_frac": float((visited & reach).sum() / reach_n),
        "beyond_wall_frac": float((visited & bw).sum() / bw_n) if bw_n else 0.0,
        "time_at_wall_frac": float(in_band.mean()) if len(zs) else 0.0,
    }
```

- [ ] **Step 4: Run to verify it passes**

Run: `cd LPM_exploration/Miniworld/experiments && PYTHONPATH=. ../../../.venv/bin/python -m pytest tests/test_coverage.py -v`
Expected: PASS (4 tests). If `test_reachable_mask` reports a count, paste that integer back into the test's regression assertion if it differs.

- [ ] **Step 5: Commit**

```bash
git add LPM_exploration/Miniworld/experiments/coverage.py LPM_exploration/Miniworld/experiments/tests/test_coverage.py
git commit -m "maze-exp: coverage geometry helpers + tests"
```

### Task 1.2: `maze_envs.py` adapter

**Files:**
- Create: `LPM_exploration/Miniworld/experiments/maze_envs.py`
- Test: `LPM_exploration/Miniworld/experiments/tests/test_maze_envs.py`

- [ ] **Step 1: Write the failing test**

```python
import numpy as np
import maze_envs


def test_make_env_each_variant_resets_and_steps():
    for variant in ["nonoise", "noisy_tv", "action_noise"]:
        env = maze_envs.make_env(variant, seed=0)
        obs, info = env.reset(seed=0)
        assert obs.shape == (120, 160, 3)
        assert obs.dtype == np.uint8
        assert "pos" in info and len(info["pos"]) == 2
        obs, r, term, trunc, info = env.step(2)  # move_forward
        assert r == 0.0
        assert "pos" in info
        env.close()


def test_obs_scale_changes_resolution():
    env = maze_envs.make_env("nonoise", seed=0, obs_scale=0.5)
    obs, _ = env.reset(seed=0)
    assert obs.shape == (60, 80, 3)
    env.close()
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd LPM_exploration/Miniworld/experiments && PYTHONPATH=. ../../../.venv/bin/python -m pytest tests/test_maze_envs.py -v`
Expected: FAIL (no `maze_envs`).

- [ ] **Step 3: Implement `maze_envs.py`**

```python
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
```

Note: `obs_scale != 1.0` with `action_noise` resizes CIFAR to the class
constant 160×120, mismatching a downscaled frame. Default is 1.0; if Phase-0
forces downscaling, restrict `action_noise` to scale 1.0 (logged limitation).

- [ ] **Step 4: Run to verify it passes**

Run: `cd LPM_exploration/Miniworld/experiments && PYTHONPATH=. ../../../.venv/bin/python -m pytest tests/test_maze_envs.py -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add LPM_exploration/Miniworld/experiments/maze_envs.py LPM_exploration/Miniworld/experiments/tests/test_maze_envs.py
git commit -m "maze-exp: env adapter reusing miniworld_play geometry"
```

---

## Phase 2 — Intrinsic-reward models

### Task 2.1: Shared encoder + decoder-prediction (MSE) model

**Files:**
- Create: `LPM_exploration/Miniworld/experiments/models.py`
- Test: `LPM_exploration/Miniworld/experiments/tests/test_models.py`

`CNNFeatureExtractor`, `A2CNetwork`, `MSEPredictionModel`,
`UncertaintyPredictionModel` are lifted **verbatim** from
`/tmp/maze_nb_with_noisyTV.txt` (lines 164-192, 298-329, 879-975, 977-1035)
except: (a) take `device` as a constructor arg instead of a module global,
(b) no behaviour change. Build the file incrementally; this task adds the
encoder + the MSE model + a uniform `IntrinsicModel` interface.

Uniform interface every method implements:
```python
class IntrinsicModel:
    def reward(self, state, next_state, action) -> float: ...
    def update(self, states, next_states, actions) -> dict: ...  # named losses
```
(`state`/`next_state` are single HWC uint8 frames; `states`/`next_states` are
torch float tensors `(N,H,W,C)`; `actions` is a LongTensor `(N,)`.)

- [ ] **Step 1: Write the failing test**

```python
import numpy as np
import torch
import models

OBS = (120, 160, 3)
INP = (3, 120, 160)


def _frame():
    return np.random.randint(0, 256, size=OBS, dtype=np.uint8)


def test_feature_extractor_shape():
    fe = models.CNNFeatureExtractor(INP)
    x = torch.rand(2, *OBS)  # (N,H,W,C)
    out = fe(x)
    assert out.shape == (2, fe.feature_size)


def test_mse_model_reward_and_update():
    m = models.MSEModel(INP, num_actions=5, device="cpu")
    r = m.reward(_frame(), _frame(), 2)
    assert isinstance(r, float) and r >= 0
    states = torch.rand(4, *OBS)
    nstates = torch.rand(4, *OBS)
    actions = torch.randint(0, 5, (4,))
    losses = m.update(states, nstates, actions)
    assert "pred_loss" in losses
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd LPM_exploration/Miniworld/experiments && PYTHONPATH=. ../../../.venv/bin/python -m pytest tests/test_models.py -v`
Expected: FAIL (no `models`).

- [ ] **Step 3: Implement encoder + MSEModel in `models.py`**

Lift `CNNFeatureExtractor` (notebook lines 164-192) verbatim. Then wrap the
notebook `MSEPredictionModel` (lines 879-975) behind the uniform interface:

```python
from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class CNNFeatureExtractor(nn.Module):
    def __init__(self, input_shape):
        super().__init__()
        self.conv1 = nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        def co(size, k, s):
            return (size - k) // s + 1
        cw = co(co(co(input_shape[2], 8, 4), 4, 2), 3, 1)
        ch = co(co(co(input_shape[1], 8, 4), 4, 2), 3, 1)
        self.feature_size = cw * ch * 64

    def forward(self, x):
        x = x / 255.0
        x = x.permute(0, 3, 1, 2)            # (N,H,W,C) -> (N,C,H,W)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        return x.reshape(x.size(0), -1)


class IntrinsicModel:
    """Uniform interface. `reward` returns a raw scalar; the A2C agent
    normalises across methods before combining."""
    def reward(self, state, next_state, action) -> float:
        raise NotImplementedError
    def update(self, states, next_states, actions) -> dict:
        return {}


class _Decoder(nn.Module):
    """Predicts next-state pixels from (features, action). Notebook decoder."""
    def __init__(self, input_shape, num_actions):
        super().__init__()
        self.num_actions = num_actions
        c, h, w = input_shape
        self.fe = CNNFeatureExtractor(input_shape)
        self.fwd = nn.Sequential(nn.Linear(self.fe.feature_size + num_actions, 512), nn.ReLU())
        ho, wo = h // 8, w // 8
        self.dec = nn.Sequential(
            nn.Linear(512, ho * wo * 64), nn.ReLU(),
            nn.Unflatten(1, (64, ho, wo)),
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(16, c, 4, 2, 1), nn.Sigmoid(),
            nn.Upsample(size=(h, w), mode="bilinear", align_corners=False),
        )

    def forward(self, state, action):
        f = self.fe(state)
        a = F.one_hot(action, self.num_actions).float()
        return self.dec(self.fwd(torch.cat([f, a], dim=1)))


class MSEModel(IntrinsicModel):
    """Raw next-state prediction error — the pure noisy-TV victim."""
    def __init__(self, input_shape, num_actions, device="cpu", lr=1e-3):
        self.device = device
        self.num_actions = num_actions
        self.net = _Decoder(input_shape, num_actions).to(device)
        self.opt = optim.Adam(self.net.parameters(), lr=lr)

    def reward(self, state, next_state, action):
        s = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        ns = torch.FloatTensor(next_state).unsqueeze(0).to(self.device)
        a = torch.LongTensor([action]).to(self.device)
        with torch.no_grad():
            pred = self.net(s, a)
            tgt = ns.permute(0, 3, 1, 2) / 255.0
            return float(((tgt - pred) ** 2).mean().item())

    def update(self, states, next_states, actions):
        pred = self.net(states, actions)
        tgt = next_states.permute(0, 3, 1, 2) / 255.0
        loss = F.mse_loss(pred, tgt)
        self.opt.zero_grad(); loss.backward(); self.opt.step()
        return {"pred_loss": float(loss.item())}
```

- [ ] **Step 4: Run to verify it passes**

Run: `cd LPM_exploration/Miniworld/experiments && PYTHONPATH=. ../../../.venv/bin/python -m pytest tests/test_models.py -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add LPM_exploration/Miniworld/experiments/models.py LPM_exploration/Miniworld/experiments/tests/test_models.py
git commit -m "maze-exp: shared CNN encoder + MSE decoder-prediction model"
```

### Task 2.2: LPM model (prediction + uncertainty)

**Files:**
- Modify: `LPM_exploration/Miniworld/experiments/models.py`
- Modify: `LPM_exploration/Miniworld/experiments/tests/test_models.py`

`LPMModel` wraps a `_Decoder` (prediction) + an uncertainty MLP, lifting
`UncertaintyPredictionModel` (notebook 977-1035) and
`LearningProgressCuriosity` (1037-1124) verbatim, behind the uniform interface.
Reward = `clip(eta*expected_err - actual_mse, max=0.5)` with eta=1.0.

- [ ] **Step 1: Add failing test**

```python
def test_lpm_model_reward_bounded_and_update():
    m = models.LPMModel(INP, num_actions=5, device="cpu")
    r = m.reward(_frame(), _frame(), 1)
    assert isinstance(r, float) and r <= 0.5
    states = torch.rand(6, *OBS); nstates = torch.rand(6, *OBS)
    actions = torch.randint(0, 5, (6,))
    losses = m.update(states, nstates, actions)
    assert "pred_loss" in losses and "unc_loss" in losses
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd LPM_exploration/Miniworld/experiments && PYTHONPATH=. ../../../.venv/bin/python -m pytest tests/test_models.py::test_lpm_model_reward_bounded_and_update -v`
Expected: FAIL (`LPMModel` missing).

- [ ] **Step 3: Implement `LPMModel` + uncertainty net in `models.py`**

```python
class _UncertaintyNet(nn.Module):
    def __init__(self, input_shape, num_actions):
        super().__init__()
        self.num_actions = num_actions
        self.fe = CNNFeatureExtractor(input_shape)
        self.net = nn.Sequential(
            nn.Linear(self.fe.feature_size + num_actions, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, 1),
        )

    def forward(self, state, action):
        f = self.fe(state)
        a = F.one_hot(action, self.num_actions).float()
        out = self.net(torch.cat([f, a], dim=1))
        return torch.clamp(out, -10.0, 10.0)


class LPMModel(IntrinsicModel):
    def __init__(self, input_shape, num_actions, device="cpu", eta=1.0,
                 buffer_size=100, update_unc_every=5):
        self.device = device
        self.num_actions = num_actions
        self.eta = eta
        self.pred = _Decoder(input_shape, num_actions).to(device)
        self.pred_opt = optim.Adam(self.pred.parameters(), lr=1e-3)
        self.unc = _UncertaintyNet(input_shape, num_actions).to(device)
        self.unc_opt = optim.Adam(self.unc.parameters(), lr=1e-2)
        self.buf = []  # (state, action, mse)
        self.buffer_size = buffer_size
        self.update_unc_every = update_unc_every
        self._since = 0

    def _mse(self, state, next_state, action):
        s = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        ns = torch.FloatTensor(next_state).unsqueeze(0).to(self.device)
        a = torch.LongTensor([action]).to(self.device)
        with torch.no_grad():
            pred = self.pred(s, a)
            tgt = ns.permute(0, 3, 1, 2) / 255.0
            return float(((tgt - pred) ** 2).mean().item()), s, a

    def reward(self, state, next_state, action):
        actual, s, a = self._mse(state, next_state, action)
        self.buf.append((state, action, actual))
        if len(self.buf) > self.buffer_size:
            self.buf.pop(0)
        with torch.no_grad():
            expected = float(torch.exp(self.unc(s, a)).item())
        return float(min(0.5, self.eta * expected - actual))

    def update(self, states, next_states, actions):
        pred = self.pred(states, actions)
        tgt = next_states.permute(0, 3, 1, 2) / 255.0
        ploss = F.mse_loss(pred, tgt)
        self.pred_opt.zero_grad(); ploss.backward(); self.pred_opt.step()
        self._since += 1
        uloss = 0.0
        if self._since >= self.update_unc_every and self.buf:
            n = min(32, len(self.buf))
            idx = np.random.choice(len(self.buf), n, replace=False)
            bs = torch.FloatTensor(np.array([self.buf[i][0] for i in idx])).to(self.device)
            ba = torch.LongTensor([self.buf[i][1] for i in idx]).to(self.device)
            be = [self.buf[i][2] for i in idx]
            logp = self.unc(bs, ba)
            logt = torch.log(torch.FloatTensor(be).to(self.device) + 1e-6).unsqueeze(1)
            ul = F.mse_loss(logp, logt)
            self.unc_opt.zero_grad(); ul.backward(); self.unc_opt.step()
            uloss = float(ul.item()); self._since = 0
        return {"pred_loss": float(ploss.item()), "unc_loss": uloss}
```

- [ ] **Step 4: Run to verify it passes**

Run: `cd LPM_exploration/Miniworld/experiments && PYTHONPATH=. ../../../.venv/bin/python -m pytest tests/test_models.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add LPM_exploration/Miniworld/experiments/models.py LPM_exploration/Miniworld/experiments/tests/test_models.py
git commit -m "maze-exp: LPM model (prediction + uncertainty)"
```

### Task 2.3: ICM (canonical) + RND + NoneModel + factory

**Files:**
- Modify: `LPM_exploration/Miniworld/experiments/models.py`
- Modify: `LPM_exploration/Miniworld/experiments/tests/test_models.py`

ICM = canonical inverse+forward dynamics on the shared encoder features
(NOT the notebook decoder). RND = fixed random target encoder + trainable
predictor. Both new; document in UPSTREAM.md.

- [ ] **Step 1: Add failing tests**

```python
import pytest

@pytest.mark.parametrize("name,expect", [
    ("icm", "fwd_loss"), ("rnd", "rnd_loss"),
])
def test_icm_rnd_reward_and_update(name, expect):
    m = models.build_model(name, INP, 5, device="cpu")
    r = m.reward(_frame(), _frame(), 3)
    assert isinstance(r, float) and r >= 0
    losses = m.update(torch.rand(4, *OBS), torch.rand(4, *OBS), torch.randint(0, 5, (4,)))
    assert expect in losses


def test_none_model_zero_reward():
    m = models.build_model("none", INP, 5, device="cpu")
    assert m.reward(_frame(), _frame(), 0) == 0.0
    assert m.update(torch.rand(2, *OBS), torch.rand(2, *OBS), torch.randint(0, 5, (2,))) == {}


def test_build_model_all_names():
    for n in ["lpm", "rnd", "icm", "mse", "none"]:
        assert isinstance(models.build_model(n, INP, 5, "cpu"), models.IntrinsicModel)
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd LPM_exploration/Miniworld/experiments && PYTHONPATH=. ../../../.venv/bin/python -m pytest tests/test_models.py -k "icm_rnd or none_model or build_model" -v`
Expected: FAIL.

- [ ] **Step 3: Implement ICM, RND, NoneModel, `build_model` in `models.py`**

```python
class ICMModel(IntrinsicModel):
    """Canonical ICM: inverse + forward dynamics in feature space."""
    def __init__(self, input_shape, num_actions, device="cpu", lr=1e-3, beta=0.2):
        self.device = device
        self.num_actions = num_actions
        self.beta = beta
        self.fe = CNNFeatureExtractor(input_shape).to(device)
        d = self.fe.feature_size
        self.inverse = nn.Sequential(nn.Linear(2 * d, 256), nn.ReLU(),
                                     nn.Linear(256, num_actions)).to(device)
        self.forward_net = nn.Sequential(nn.Linear(d + num_actions, 256), nn.ReLU(),
                                         nn.Linear(256, d)).to(device)
        params = (list(self.fe.parameters()) + list(self.inverse.parameters())
                  + list(self.forward_net.parameters()))
        self.opt = optim.Adam(params, lr=lr)

    def _phi(self, x):
        return self.fe(x)

    def reward(self, state, next_state, action):
        s = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        ns = torch.FloatTensor(next_state).unsqueeze(0).to(self.device)
        a = torch.LongTensor([action]).to(self.device)
        with torch.no_grad():
            phi, nphi = self._phi(s), self._phi(ns)
            ah = F.one_hot(a, self.num_actions).float()
            pred = self.forward_net(torch.cat([phi, ah], dim=1))
            return float(((pred - nphi) ** 2).mean().item())

    def update(self, states, next_states, actions):
        phi, nphi = self._phi(states), self._phi(next_states)
        ah = F.one_hot(actions, self.num_actions).float()
        pred = self.forward_net(torch.cat([phi, ah], dim=1))
        fwd_loss = F.mse_loss(pred, nphi.detach())
        logits = self.inverse(torch.cat([phi, nphi], dim=1))
        inv_loss = F.cross_entropy(logits, actions)
        loss = self.beta * fwd_loss + (1 - self.beta) * inv_loss
        self.opt.zero_grad(); loss.backward(); self.opt.step()
        return {"fwd_loss": float(fwd_loss.item()), "inv_loss": float(inv_loss.item())}


class RNDModel(IntrinsicModel):
    """Random Network Distillation: predictor matches a frozen random target."""
    def __init__(self, input_shape, num_actions, device="cpu", lr=1e-3, emb=256):
        self.device = device
        self.target = CNNFeatureExtractor(input_shape).to(device)
        d = self.target.feature_size
        self.target_head = nn.Linear(d, emb).to(device)
        for p in list(self.target.parameters()) + list(self.target_head.parameters()):
            p.requires_grad_(False)
        self.pred = CNNFeatureExtractor(input_shape).to(device)
        self.pred_head = nn.Linear(d, emb).to(device)
        self.opt = optim.Adam(list(self.pred.parameters()) + list(self.pred_head.parameters()), lr=lr)

    def _t(self, x):
        return self.target_head(self.target(x))

    def _p(self, x):
        return self.pred_head(self.pred(x))

    def reward(self, state, next_state, action):
        ns = torch.FloatTensor(next_state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            return float(((self._p(ns) - self._t(ns)) ** 2).mean().item())

    def update(self, states, next_states, actions):
        with torch.no_grad():
            tgt = self._t(next_states)
        loss = F.mse_loss(self._p(next_states), tgt)
        self.opt.zero_grad(); loss.backward(); self.opt.step()
        return {"rnd_loss": float(loss.item())}


class NoneModel(IntrinsicModel):
    def __init__(self, *a, **k):
        pass
    def reward(self, state, next_state, action):
        return 0.0
    def update(self, states, next_states, actions):
        return {}


def build_model(name, input_shape, num_actions, device="cpu"):
    name = name.lower()
    if name == "lpm":
        return LPMModel(input_shape, num_actions, device)
    if name == "icm":
        return ICMModel(input_shape, num_actions, device)
    if name == "rnd":
        return RNDModel(input_shape, num_actions, device)
    if name == "mse":
        return MSEModel(input_shape, num_actions, device)
    if name == "none":
        return NoneModel()
    raise ValueError(f"unknown method {name!r}")
```

- [ ] **Step 4: Run to verify it passes**

Run: `cd LPM_exploration/Miniworld/experiments && PYTHONPATH=. ../../../.venv/bin/python -m pytest tests/test_models.py -v`
Expected: PASS (all model tests).

- [ ] **Step 5: Commit**

```bash
git add LPM_exploration/Miniworld/experiments/models.py LPM_exploration/Miniworld/experiments/tests/test_models.py
git commit -m "maze-exp: canonical ICM + RND + none baseline + factory"
```

---

## Phase 3 — A2C agent

### Task 3.1: `a2c.py` (generalized agent with intrinsic normalization + position hook)

**Files:**
- Create: `LPM_exploration/Miniworld/experiments/a2c.py`
- Test: `LPM_exploration/Miniworld/experiments/tests/test_a2c.py`

Lift `A2CNetwork` (notebook 298-329), `Memory` (331-363), and the agent's
`select_action` / `compute_gae` / `update` (1265-1363) **verbatim**, generalized:
the agent takes any `IntrinsicModel`; the reward combine is uniform:
`combined = lambda_intrinsic * normalize(intrinsic)` (running mean/std,
RND-style), dropping the notebook's ad-hoc offsets. The training loop lives in
`train_maze.py` (Task 4.1), not here, so the agent stays unit-testable.

- [ ] **Step 1: Write the failing test**

```python
import numpy as np
import torch
import a2c, models

OBS = (120, 160, 3)


def test_running_norm_converges_to_unit_std():
    rn = a2c.RunningNorm()
    for v in np.random.randn(500) * 3 + 7:
        rn.update(float(v))
    x = rn.normalize(10.0)
    assert abs(x) < 5  # finite, roughly standardized


def test_agent_select_and_update_runs():
    net = a2c.A2CNetwork((3, 120, 160), 5).to("cpu")
    agent = a2c.A2CAgent(net, num_actions=5, device="cpu", lambda_intrinsic=0.1)
    state = np.random.randint(0, 256, OBS, dtype=np.uint8)
    a, lp, v = agent.select_action(state)
    assert 0 <= a < 5
    # Fill memory with a few fake transitions, then update.
    for _ in range(8):
        ns = np.random.randint(0, 256, OBS, dtype=np.uint8)
        agent.memory.add(state, a, 0.0, 0.01, ns, False, lp, v)
    losses = agent.update()
    assert "policy_loss" in losses and "value_loss" in losses
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd LPM_exploration/Miniworld/experiments && PYTHONPATH=. ../../../.venv/bin/python -m pytest tests/test_a2c.py -v`
Expected: FAIL (no `a2c`).

- [ ] **Step 3: Implement `a2c.py`**

```python
"""A2C actor-critic extracted from the maze notebooks (verbatim policy/update),
generalized to any IntrinsicModel with uniform running-std reward normalization.
"""
from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical


class A2CNetwork(nn.Module):
    def __init__(self, input_shape, num_actions):
        super().__init__()
        from models import CNNFeatureExtractor
        self.feature_extractor = CNNFeatureExtractor(input_shape)
        fs = self.feature_extractor.feature_size
        self.fc_actor = nn.Sequential(nn.Linear(fs, 512), nn.ReLU(), nn.Linear(512, num_actions))
        self.fc_critic = nn.Sequential(nn.Linear(fs, 512), nn.ReLU(), nn.Linear(512, 1))

    def forward(self, x):
        f = self.feature_extractor(x)
        return self.fc_actor(f), self.fc_critic(f)


class Memory:
    def __init__(self):
        self.clear()
    def add(self, s, a, r, ir, ns, d, lp, v):
        self.states.append(s); self.actions.append(a); self.rewards.append(r)
        self.intrinsic_rewards.append(ir); self.next_states.append(ns)
        self.dones.append(d); self.logprobs.append(lp); self.values.append(v)
    def clear(self):
        self.states, self.actions, self.rewards = [], [], []
        self.intrinsic_rewards, self.next_states, self.dones = [], [], []
        self.logprobs, self.values = [], []
    def __len__(self):
        return len(self.states)


class RunningNorm:
    """Welford running mean/std for scalar intrinsic rewards."""
    def __init__(self):
        self.n = 0; self.mean = 0.0; self.m2 = 0.0
    def update(self, x):
        self.n += 1
        d = x - self.mean
        self.mean += d / self.n
        self.m2 += d * (x - self.mean)
    @property
    def std(self):
        return (self.m2 / self.n) ** 0.5 if self.n > 1 else 1.0
    def normalize(self, x):
        return (x - self.mean) / (self.std + 1e-8)


class A2CAgent:
    def __init__(self, policy, num_actions, device="cpu", gamma=0.99, gae_lambda=0.95,
                 lambda_intrinsic=0.1, entropy_coef=0.05, value_loss_coef=0.5,
                 max_grad_norm=0.5, lr=0.01, normalize_intrinsic=True):
        self.policy = policy
        self.num_actions = num_actions
        self.device = device
        self.gamma = gamma; self.gae_lambda = gae_lambda
        self.lambda_intrinsic = lambda_intrinsic
        self.entropy_coef = entropy_coef; self.value_loss_coef = value_loss_coef
        self.max_grad_norm = max_grad_norm
        self.optimizer = optim.RMSprop(policy.parameters(), lr=lr, alpha=0.99, eps=1e-8)
        self.memory = Memory()
        self.rn = RunningNorm()
        self.normalize_intrinsic = normalize_intrinsic

    def select_action(self, state):
        st = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits, value = self.policy(st)
            dist = Categorical(F.softmax(logits, dim=-1))
            a = dist.sample()
            return int(a.item()), float(dist.log_prob(a).item()), float(value.item())

    def compute_gae(self, rewards, values, dones, next_value=0.0):
        returns, gae = [], 0.0
        for i in reversed(range(len(rewards))):
            delta = rewards[i] + self.gamma * next_value * (1 - dones[i]) - values[i]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[i]) * gae
            next_value = values[i]
            returns.insert(0, gae + values[i])
        return returns

    def update(self):
        if len(self.memory) == 0:
            return {}
        states = torch.FloatTensor(np.array(self.memory.states)).to(self.device)
        actions = torch.LongTensor(self.memory.actions).to(self.device)
        dones = np.array(self.memory.dones, dtype=float)
        ext = np.array(self.memory.rewards, dtype=float)
        intr = np.array(self.memory.intrinsic_rewards, dtype=float)
        if self.normalize_intrinsic:
            for v in intr:
                self.rn.update(float(v))
            intr = np.array([self.rn.normalize(float(v)) for v in intr])
        combined = ext + self.lambda_intrinsic * intr
        with torch.no_grad():
            last = torch.FloatTensor(self.memory.next_states[-1]).unsqueeze(0).to(self.device)
            _, lv = self.policy(last); last_value = float(lv.item())
        returns = torch.FloatTensor(
            self.compute_gae(combined, self.memory.values, dones, last_value)).to(self.device)
        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        logits, values = self.policy(states)
        dist = Categorical(F.softmax(logits, dim=-1))
        adv = returns - values.squeeze()
        policy_loss = -(dist.log_prob(actions) * adv.detach()).mean()
        value_loss = F.mse_loss(values.squeeze(), returns)
        entropy = dist.entropy().mean()
        loss = policy_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy
        self.optimizer.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
        self.optimizer.step()
        self.memory.clear()
        return {"policy_loss": float(policy_loss.item()),
                "value_loss": float(value_loss.item()),
                "entropy": float(entropy.item())}
```

- [ ] **Step 4: Run to verify it passes**

Run: `cd LPM_exploration/Miniworld/experiments && PYTHONPATH=. ../../../.venv/bin/python -m pytest tests/test_a2c.py -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add LPM_exploration/Miniworld/experiments/a2c.py LPM_exploration/Miniworld/experiments/tests/test_a2c.py
git commit -m "maze-exp: A2C agent with uniform intrinsic-reward normalization"
```

---

## Phase 4 — Trainer CLI

### Task 4.1: `train_maze.py` (one run → CSV + position npz)

**Files:**
- Create: `LPM_exploration/Miniworld/experiments/train_maze.py`
- Test: `LPM_exploration/Miniworld/experiments/tests/test_train_smoke.py`

The training loop (lifted from notebook 1407-1477) collects rollouts of length
`--update-frequency` (default 64), computes intrinsic reward each step, records
`(step, x, z, action, sticky)`, and writes a CSV row per update with coverage
metrics computed from the accumulated positions. Position arrays saved to
`--pos-log` `.npz` at the end.

- [ ] **Step 1: Write the failing smoke test**

```python
import os, subprocess, sys, tempfile, numpy as np, csv

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.dirname(HERE)
PY = os.path.join(EXP, "..", "..", ".venv", "bin", "python")


def test_train_smoke_none_method(tmp_path):
    csv_path = tmp_path / "smoke.csv"
    pos_path = tmp_path / "smoke.npz"
    env = dict(os.environ, PYTHONPATH=EXP, PYTORCH_ENABLE_MPS_FALLBACK="1")
    cmd = [PY, os.path.join(EXP, "train_maze.py"), "--method", "none",
           "--variant", "noisy_tv", "--seed", "0", "--steps", "128",
           "--update-frequency", "64", "--device", "cpu",
           "--csv-log", str(csv_path), "--pos-log", str(pos_path),
           "--log-interval", "1"]
    r = subprocess.run(cmd, env=env, cwd=EXP, capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stderr[-2000:]
    with open(csv_path) as f:
        rows = list(csv.DictReader(f))
    assert len(rows) >= 1
    assert float(rows[-1]["coverage_frac"]) > 0
    d = np.load(pos_path)
    assert len(d["x"]) == 128 and len(d["z"]) == 128
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd LPM_exploration/Miniworld/experiments && PYTHONPATH=. ../../../.venv/bin/python -m pytest tests/test_train_smoke.py -v`
Expected: FAIL (no `train_maze.py`).

- [ ] **Step 3: Implement `train_maze.py`**

```python
"""Train one A2C+intrinsic run on a maze variant; log per-update metrics + positions.

Example:
  python train_maze.py --method lpm --variant noisy_tv --seed 1 --steps 20000 \
      --csv-log results/lpm-noisy_tv-s1.csv --pos-log positions/lpm-noisy_tv-s1.npz
"""
from __future__ import annotations
import argparse, csv, os, random, time
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", choices=["lpm", "rnd", "icm", "mse", "none"], required=True)
    ap.add_argument("--variant", choices=["nonoise", "noisy_tv", "action_noise"], required=True)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--update-frequency", type=int, default=64)
    ap.add_argument("--lambda-intrinsic", type=float, default=0.1)
    ap.add_argument("--obs-scale", type=float, default=1.0)
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
                     lambda_intrinsic=args.lambda_intrinsic)

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
    xs.append(info["pos"][0]); zs.append(info["pos"][1]); acts.append(-1); sticky.append(False)

    update_i = 0; t0 = time.time(); int_accum = []
    for step in range(1, args.steps + 1):
        a, lp, v = agent.select_action(state)
        ns, r, term, trunc, info = env.step(a)
        done = term or trunc
        ir = model.reward(state, ns, a)
        int_accum.append(ir)
        agent.memory.add(state, a, r, ir, ns, done, lp, v)
        xs.append(info["pos"][0]); zs.append(info["pos"][1])
        acts.append(int(info.get("action_id", a))); sticky.append(bool(info.get("sticky_replayed", False)))
        state = ns

        if step % args.update_frequency == 0:
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
```

- [ ] **Step 4: Run to verify it passes**

Run: `cd LPM_exploration/Miniworld/experiments && PYTHONPATH=. ../../../.venv/bin/python -m pytest tests/test_train_smoke.py -v`
Expected: PASS (takes a minute — 128 steps with no intrinsic model).

- [ ] **Step 5: Smoke-test the LPM path manually (intrinsic forward each step)**

Run: `cd LPM_exploration/Miniworld/experiments && PYTHONPATH=. PYTORCH_ENABLE_MPS_FALLBACK=1 ../../../.venv/bin/python train_maze.py --method lpm --variant noisy_tv --seed 0 --steps 128 --update-frequency 64 --device cpu --csv-log /tmp/lpm_smoke.csv --pos-log /tmp/lpm_smoke.npz --log-interval 1`
Expected: prints `[done] lpm-noisy_tv-s0: coverage={...}` with no error. **Record steps/s** from the fps column to finalize the grid budget.

- [ ] **Step 6: Commit**

```bash
git add LPM_exploration/Miniworld/experiments/train_maze.py LPM_exploration/Miniworld/experiments/tests/test_train_smoke.py
git commit -m "maze-exp: train_maze.py CLI with CSV + position logging"
```

---

## Phase 5 — Grid runner

### Task 5.1: `run_grid.py`

**Files:**
- Create: `LPM_exploration/Miniworld/experiments/run_grid.py`

Mirrors `Atari/experiments/run_grid.py`: iterate method × variant × seed,
build `train_maze.py` commands, run sequentially, resume by skipping CSVs with
enough rows.

- [ ] **Step 1: Implement `run_grid.py`**

```python
"""Grid runner for the maze exploration comparison.

Usage: python run_grid.py --steps 20000 --seeds 1 2  [--dry-run]
"""
import argparse, csv, itertools, os, subprocess, sys

EXP = os.path.dirname(os.path.abspath(__file__))
PY = os.path.join(os.path.dirname(os.path.dirname(EXP)), ".venv", "bin", "python")
RESULTS = os.path.join(EXP, "results")
POSITIONS = os.path.join(EXP, "positions")

METHODS = ["lpm", "rnd", "icm", "mse", "none"]
VARIANTS = ["nonoise", "noisy_tv", "action_noise"]


def enough_rows(path, n=2):
    if not os.path.exists(path):
        return False
    with open(path) as f:
        return sum(1 for _ in f) > n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2])
    ap.add_argument("--methods", nargs="+", default=METHODS)
    ap.add_argument("--variants", nargs="+", default=VARIANTS)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--obs-scale", type=float, default=1.0)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    os.makedirs(RESULTS, exist_ok=True); os.makedirs(POSITIONS, exist_ok=True)
    os.makedirs("/tmp/maze_logs", exist_ok=True)
    env = dict(os.environ, PYTHONPATH=EXP, PYTORCH_ENABLE_MPS_FALLBACK="1")

    runs = list(itertools.product(args.methods, args.variants, args.seeds))
    print(f"{len(runs)} runs planned")
    for method, variant, seed in runs:
        rid = f"{method}-{variant}-s{seed}"
        csv_path = os.path.join(RESULTS, rid + ".csv")
        if enough_rows(csv_path):
            print(f"[skip] {rid} (already has results)"); continue
        cmd = [PY, os.path.join(EXP, "train_maze.py"), "--method", method,
               "--variant", variant, "--seed", str(seed), "--steps", str(args.steps),
               "--device", args.device, "--obs-scale", str(args.obs_scale),
               "--csv-log", csv_path,
               "--pos-log", os.path.join(POSITIONS, rid + ".npz"),
               "--log-interval", "5"]
        print(f"[run] {rid}: {' '.join(cmd)}")
        if args.dry_run:
            continue
        with open(f"/tmp/maze_logs/{rid}.out", "w") as out:
            res = subprocess.run(cmd, env=env, cwd=EXP, stdout=out, stderr=subprocess.STDOUT)
        if res.returncode != 0:
            print(f"[FAIL] {rid} (see /tmp/maze_logs/{rid}.out)")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Dry-run to verify the grid**

Run: `cd LPM_exploration/Miniworld/experiments && ../../../.venv/bin/python run_grid.py --dry-run --seeds 1 2`
Expected: prints "30 runs planned" and 30 `[run]` command lines.

- [ ] **Step 3: Commit**

```bash
git add LPM_exploration/Miniworld/experiments/run_grid.py
git commit -m "maze-exp: grid runner (method x variant x seed) with resume"
```

---

## Phase 6 — Analysis + heatmaps

### Task 6.1: `heatmaps.py` (binning + maze overlay)

**Files:**
- Create: `LPM_exploration/Miniworld/experiments/heatmaps.py`
- Test: `LPM_exploration/Miniworld/experiments/tests/test_heatmaps.py`

- [ ] **Step 1: Write the failing test**

```python
import numpy as np
import heatmaps


def test_window_occupancy_shapes():
    steps = np.arange(1000)
    xs = np.random.uniform(0, 18, 1000)
    zs = np.random.uniform(0, 12, 1000)
    occ = heatmaps.window_occupancy(steps, xs, zs, n_windows=5)
    assert occ.shape == (5, 72, 48)
    assert occ.sum() == 1000


def test_cumulative_frontier_is_monotone():
    steps = np.arange(1000)
    xs = np.random.uniform(0, 18, 1000)
    zs = np.random.uniform(0, 12, 1000)
    fro = heatmaps.cumulative_frontier(steps, xs, zs, n_windows=5)
    counts = [(fro[i] > 0).sum() for i in range(5)]
    assert counts == sorted(counts)  # coverage only grows
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd LPM_exploration/Miniworld/experiments && PYTHONPATH=. ../../../.venv/bin/python -m pytest tests/test_heatmaps.py -v`
Expected: FAIL (no `heatmaps`).

- [ ] **Step 3: Implement `heatmaps.py`**

```python
"""Occupancy/coverage heatmaps over training-progress windows, with maze overlay."""
from __future__ import annotations
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import coverage as cov


def _bin(xs, zs):
    ix = np.clip((np.asarray(xs) * 4).astype(int), 0, cov.NX - 1)
    iz = np.clip((np.asarray(zs) * 4).astype(int), 0, cov.NZ - 1)
    return ix, iz


def window_occupancy(steps, xs, zs, n_windows=5):
    """(n_windows, NX, NZ) step-count per cell within each step window."""
    steps = np.asarray(steps)
    edges = np.linspace(steps.min(), steps.max() + 1, n_windows + 1)
    occ = np.zeros((n_windows, cov.NX, cov.NZ), dtype=np.int64)
    ix, iz = _bin(xs, zs)
    w = np.clip(np.searchsorted(edges, steps, side="right") - 1, 0, n_windows - 1)
    for k in range(n_windows):
        sel = w == k
        np.add.at(occ[k], (ix[sel], iz[sel]), 1)
    return occ


def cumulative_frontier(steps, xs, zs, n_windows=5):
    """(n_windows, NX, NZ) binary: cells ever visited up to end of each window."""
    occ = window_occupancy(steps, xs, zs, n_windows)
    fro = np.zeros_like(occ)
    seen = np.zeros((cov.NX, cov.NZ), dtype=bool)
    for k in range(n_windows):
        seen = seen | (occ[k] > 0)
        fro[k] = seen.astype(np.int64)
    return fro


def _overlay(ax):
    # Cells are (ix, iz); imshow with origin lower, extent in world units.
    for x0, x1, z0, z1 in cov.ROOMS[:2] + [cov.ROOMS[3]]:
        ax.add_patch(Rectangle((x0, z0), x1 - x0, z1 - z0, fill=False, ec="white", lw=0.8))
    ax.axhline(cov.WALL_Z, color="red", lw=1.0, ls="--")
    ax.set_xlim(0, 18); ax.set_ylim(0, 12)


def plot_evolution(per_method, variant, out_path, mode="density", n_windows=5):
    """per_method: dict method -> (steps, xs, zs). Saves a rows×windows grid."""
    methods = list(per_method)
    fig, axes = plt.subplots(len(methods), n_windows,
                             figsize=(2.2 * n_windows, 2.0 * len(methods)), squeeze=False)
    for r, m in enumerate(methods):
        steps, xs, zs = per_method[m]
        grid = (window_occupancy(steps, xs, zs, n_windows) if mode == "density"
                else cumulative_frontier(steps, xs, zs, n_windows))
        for c in range(n_windows):
            ax = axes[r][c]
            data = grid[c].T  # (NZ, NX) so z is vertical
            disp = np.log1p(data) if mode == "density" else data
            ax.imshow(disp, origin="lower", extent=[0, 18, 0, 12],
                      aspect="auto", cmap="viridis")
            _overlay(ax)
            if r == 0:
                ax.set_title(f"win {c+1}/{n_windows}", fontsize=8)
            if c == 0:
                ax.set_ylabel(m, fontsize=9)
            ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(f"{variant} — {'occupancy density' if mode=='density' else 'coverage frontier'}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130); plt.close(fig)
```

- [ ] **Step 4: Run to verify it passes**

Run: `cd LPM_exploration/Miniworld/experiments && PYTHONPATH=. ../../../.venv/bin/python -m pytest tests/test_heatmaps.py -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add LPM_exploration/Miniworld/experiments/heatmaps.py LPM_exploration/Miniworld/experiments/tests/test_heatmaps.py
git commit -m "maze-exp: heatmap binning + maze-overlay plotting"
```

### Task 6.2: `analyze.py` (table + curves + heatmap figures)

**Files:**
- Create: `LPM_exploration/Miniworld/experiments/analyze.py`
- Test: `LPM_exploration/Miniworld/experiments/tests/test_analyze.py`

- [ ] **Step 1: Write the failing test** (synthetic results dir → expect outputs)

```python
import os, csv, numpy as np, analyze


def _fake_run(results, positions, rid):
    with open(os.path.join(results, rid + ".csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["update", "frames", "coverage_frac", "beyond_wall_frac", "time_at_wall_frac"])
        for u in range(1, 11):
            w.writerow([u, u * 64, 0.01 * u, 0.005 * u, 0.3])
    n = 640
    np.savez_compressed(os.path.join(positions, rid + ".npz"),
                        step=np.arange(n, dtype=np.int32),
                        x=np.random.uniform(0, 18, n).astype(np.float32),
                        z=np.random.uniform(0, 12, n).astype(np.float32))


def test_analyze_produces_table_and_figures(tmp_path):
    results = tmp_path / "results"; positions = tmp_path / "positions"
    figures = tmp_path / "figures"
    results.mkdir(); positions.mkdir()
    for m in ["lpm", "rnd"]:
        for v in ["nonoise", "noisy_tv"]:
            _fake_run(str(results), str(positions), f"{m}-{v}-s1")
    analyze.run(str(results), str(positions), str(figures))
    assert (figures / "table_coverage.csv").exists()
    assert (figures / "fig_coverage_curves.png").exists()
    assert (figures / "fig_beyond_wall.png").exists()
    assert (figures / "fig_heatmap_evolution_noisy_tv_density.png").exists()
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd LPM_exploration/Miniworld/experiments && PYTHONPATH=. ../../../.venv/bin/python -m pytest tests/test_analyze.py -v`
Expected: FAIL (no `analyze`).

- [ ] **Step 3: Implement `analyze.py`**

```python
"""Aggregate maze runs into a coverage table + curves + heatmap-evolution figures.

Usage: python analyze.py  (uses ./results, ./positions -> ./figures)
"""
from __future__ import annotations
import argparse, glob, os, re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import heatmaps

RID = re.compile(r"(?P<method>\w+)-(?P<variant>nonoise|noisy_tv|action_noise)-s(?P<seed>\d+)")


def _final(df, col, frac=0.1):
    k = max(1, int(len(df) * frac))
    return df[col].iloc[-k:].mean()


def run(results_dir, positions_dir, figures_dir, n_windows=5):
    os.makedirs(figures_dir, exist_ok=True)
    rows = []
    for path in sorted(glob.glob(os.path.join(results_dir, "*.csv"))):
        m = RID.search(os.path.basename(path))
        if not m:
            continue
        df = pd.read_csv(path)
        if df.empty:
            continue
        rows.append({
            "method": m["method"], "variant": m["variant"], "seed": int(m["seed"]),
            "coverage_frac": _final(df, "coverage_frac"),
            "beyond_wall_frac": _final(df, "beyond_wall_frac"),
            "time_at_wall_frac": _final(df, "time_at_wall_frac"),
            "_df": df,
        })
    if not rows:
        raise SystemExit("no runs found")
    data = pd.DataFrame([{k: v for k, v in r.items() if k != "_df"} for r in rows])

    # --- table ---
    tbl = data.groupby(["method", "variant"]).agg(["mean", "std"]).round(4)
    tbl.to_csv(os.path.join(figures_dir, "table_coverage.csv"))

    variants = sorted(data["variant"].unique())
    methods = sorted(data["method"].unique())

    # --- coverage curves (one subplot per variant) ---
    fig, axes = plt.subplots(1, len(variants), figsize=(5 * len(variants), 4), squeeze=False)
    for j, v in enumerate(variants):
        ax = axes[0][j]
        for meth in methods:
            curves = [r["_df"] for r in rows if r["method"] == meth and r["variant"] == v]
            if not curves:
                continue
            grid = np.linspace(0, max(c["frames"].max() for c in curves), 100)
            ys = [np.interp(grid, c["frames"], c["coverage_frac"]) for c in curves]
            mean = np.mean(ys, axis=0); std = np.std(ys, axis=0)
            ax.plot(grid, mean, label=meth)
            ax.fill_between(grid, mean - std, mean + std, alpha=0.15)
        ax.set_title(v); ax.set_xlabel("frames"); ax.set_ylabel("coverage_frac"); ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(os.path.join(figures_dir, "fig_coverage_curves.png"), dpi=130); plt.close(fig)

    # --- beyond-wall bar (headline robustness) ---
    piv = data.groupby(["method", "variant"])["beyond_wall_frac"].mean().unstack("variant")
    fig, ax = plt.subplots(figsize=(7, 4))
    piv.plot(kind="bar", ax=ax); ax.set_ylabel("beyond_wall_frac (final)")
    ax.set_title("Coverage past the noise wall (room4)")
    fig.tight_layout(); fig.savefig(os.path.join(figures_dir, "fig_beyond_wall.png"), dpi=130); plt.close(fig)

    # --- time-at-wall bar ---
    piv2 = data.groupby(["method", "variant"])["time_at_wall_frac"].mean().unstack("variant")
    fig, ax = plt.subplots(figsize=(7, 4))
    piv2.plot(kind="bar", ax=ax); ax.set_ylabel("time_at_wall_frac")
    ax.set_title("Fraction of steps lingering at the noise wall")
    fig.tight_layout(); fig.savefig(os.path.join(figures_dir, "fig_time_at_wall.png"), dpi=130); plt.close(fig)

    # --- heatmap evolution (one fig per variant per mode), seed 1 if present ---
    for v in variants:
        per_method = {}
        for meth in methods:
            cand = [r for r in rows if r["method"] == meth and r["variant"] == v]
            if not cand:
                continue
            seed = min(c["seed"] for c in cand)
            npz = os.path.join(positions_dir, f"{meth}-{v}-s{seed}.npz")
            if not os.path.exists(npz):
                continue
            d = np.load(npz)
            per_method[meth] = (d["step"], d["x"], d["z"])
        if not per_method:
            continue
        for mode in ("density", "frontier"):
            out = os.path.join(figures_dir, f"fig_heatmap_evolution_{v}_{mode}.png")
            heatmaps.plot_evolution(per_method, v, out, mode=mode, n_windows=n_windows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--positions", default="positions")
    ap.add_argument("--figures", default="figures")
    ap.add_argument("--n-windows", type=int, default=5)
    a = ap.parse_args()
    run(a.results, a.positions, a.figures, a.n_windows)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run to verify it passes**

Run: `cd LPM_exploration/Miniworld/experiments && PYTHONPATH=. ../../../.venv/bin/python -m pytest tests/test_analyze.py -v`
Expected: PASS (1 test). (Confirms `pandas` is available; if not, `uv pip install --python ../../.venv pandas`.)

- [ ] **Step 5: Commit**

```bash
git add LPM_exploration/Miniworld/experiments/analyze.py LPM_exploration/Miniworld/experiments/tests/test_analyze.py
git commit -m "maze-exp: analyze.py -> coverage table + curves + heatmap figures"
```

---

## Phase 7 — Run the downscaled grid

### Task 7.1: Calibrate budget, run grid, analyze

- [ ] **Step 1: Finalize budget** from the Task 4.1 Step-5 steps/s figure. Target
  ≈ a few hours total for 30 runs. If `S` steps/s for the slowest method (LPM),
  pick `--steps` so `30 * steps / S` is acceptable; record the choice.

- [ ] **Step 2: Run the grid** (background; long-running)

Run: `cd LPM_exploration/Miniworld/experiments && PYTORCH_ENABLE_MPS_FALLBACK=1 ../../../.venv/bin/python run_grid.py --steps <BUDGET> --seeds 1 2`
Expected: per-run `[run]` lines; CSVs appear in `results/`, npz in `positions/`. Resumable.

- [ ] **Step 3: Analyze**

Run: `cd LPM_exploration/Miniworld/experiments && PYTHONPATH=. ../../../.venv/bin/python analyze.py`
Expected: `figures/` populated with the table + curves + per-variant heatmap-evolution PNGs.

- [ ] **Step 4: Sanity-check results** — `coverage_frac` rises over training; under
  `noisy_tv`, MSE (and likely ICM/RND) show lower `beyond_wall_frac` and higher
  `time_at_wall_frac` than LPM. Note deviations from the hypothesis for the write-up.

---

## Phase 8 — Write-up, docs, gitignore

### Task 8.1: `.gitignore` + UPSTREAM.md + README

**Files:**
- Modify: `.gitignore`
- Modify: `LPM_exploration/UPSTREAM.md`
- Modify: `README.md`

- [ ] **Step 1: Append to `.gitignore`**

```
# Maze exploration experiment artifacts (regenerable)
LPM_exploration/Miniworld/experiments/results/
LPM_exploration/Miniworld/experiments/positions/
LPM_exploration/Miniworld/experiments/figures/
```

- [ ] **Step 2: Add a "Local additions / deviations" note to `UPSTREAM.md`**
  covering: A2C + models extracted into `Miniworld/experiments/`; canonical ICM
  and RND newly added (absent from notebooks); uniform running-std intrinsic
  normalization replacing the notebook's ad-hoc offsets; env reused from
  `miniworld_play/envs.py`; any `--steps`/`--obs-scale` downscaling.

- [ ] **Step 3: Add a short README section** pointing to the experiment
  (`run_grid.py` → `analyze.py`) and the new heatmap statistic.

- [ ] **Step 4: Commit**

```bash
git add .gitignore LPM_exploration/UPSTREAM.md README.md
git commit -m "maze-exp: gitignore artifacts + UPSTREAM/README notes"
```

### Task 8.2: LaTeX design write-up

**Files:**
- Create: `latex_notes/2026-05-31-maze-exploration-design.tex`

- [ ] **Step 1: Write the `.tex`** mirroring
  `latex_notes/2026-05-31-pacman-exploration-design.tex`: Motivation + RQ1/RQ2;
  Methods table (LPM/RND/ICM/MSE/none + the normalization note); Experimental
  design (5×3×seeds grid, single-env A2C, budget from Phase-0); Metrics
  (coverage_frac, beyond_wall_frac, time_at_wall_frac + the heatmap-evolution
  statistic); Infrastructure (reuses `miniworld_play` env, mirrors Ms Pac-Man
  infra); Limitations (downscaled budget, single seed-pair, ICM/RND ports,
  obs-scale). Use real LaTeX math here (this file is the documented exception to
  the terminal linear-notation rule). Embed the generated figures.

- [ ] **Step 2: Compile to PDF**

Run: `cd latex_notes && pdflatex -interaction=nonstopmode 2026-05-31-maze-exploration-design.tex`
Expected: PDF produced (warnings ok). If a figure is missing, include only the figures that exist.

- [ ] **Step 3: Commit**

```bash
git add latex_notes/2026-05-31-maze-exploration-design.tex latex_notes/2026-05-31-maze-exploration-design.pdf
git commit -m "latex_notes: 3D-maze exploration-comparison experiment design"
```

---

## Self-review notes

- **Spec coverage:** RQ1/RQ2 → Tasks 6.2/6.1; methods LPM/RND/ICM/MSE/none →
  Tasks 2.1-2.3; 3 variants → Task 1.2; A2C engine → Task 3.1; position logging
  → Task 4.1; grid → Task 5.1; heatmap statistic → Task 6.1; metrics
  (coverage/beyond_wall/time_at_wall) → Task 1.1 + 6.2; Phase-0 calibration →
  Tasks 0.2/4.1; deviations log → Task 8.1; latex write-up → Task 8.2. No gaps.
- **Type consistency:** `IntrinsicModel.reward/update` uniform across all models;
  `build_model` names match `train_maze`/`run_grid` choices
  (`lpm/rnd/icm/mse/none`); CSV `cols` superset covers every model's loss keys
  (`pred_loss/unc_loss/fwd_loss/rnd_loss`) + agent losses
  (`policy_loss/value_loss/entropy`); `coverage_metrics` keys match the CSV
  columns written in `train_maze`.
- **Deviation flagged:** the notebook fed obs as `(N,H,W,C)` and the encoder
  permutes internally — preserved exactly so extracted weights/behaviour match.
