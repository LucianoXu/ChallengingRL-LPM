# MiniGrid Intrinsic-Reward Rebuild Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Incorporate Youssef's MiniGrid intrinsic-reward project into this repo as `minigrid_exp/`, add a paper-faithful LPM intrinsic-reward wrapper alongside the existing RND one, and wire a process-parallel experiment grid + analysis that answers the SPEC's β-sweep, sparse-reward, and LPM-vs-RND-under-noise questions.

**Architecture:** Vendor the source from the private repo `JosefGh/minigrid_intrinsic_reward` (SB3 DQN+UCB / PPO on flat MiniGrid observations) into a non-shadowing top-level package `minigrid_exp/`. Keep the existing global-observation-noise wrapper and RND wrapper. Add `LPMIntrinsicRewardWrapper` (MLP/flat-obs port of our validated log-space `LPMModel`) selectable via `method`. Make `make_env`/`train_agent` accept the swept dimensions (`method`, `beta`) as explicit parameters, then drive everything from a `run_grid.py` that launches `train_one.py` subprocesses (process-level parallelism, like the maze `run_grid.py`), writing all artifacts under `expr_data/minigrid/`.

**Tech Stack:** Python 3.11 (the existing `LPM_exploration/.venv`), `gymnasium`, `minigrid`, `stable-baselines3`, `torch` (CPU), `numpy`, `pandas`, `matplotlib`.

## Global Constraints

- **Package directory is `minigrid_exp/`**, NOT `minigrid/` — a top-level `minigrid/` would shadow the installed `minigrid` pip package on `import minigrid`.
- **All experiment artifacts go under `expr_data/minigrid/`** (gitignored), never inside the package tree. Subdirs: `expr_data/minigrid/{results,models,logs,figures}`.
- **8 seeds** per experiment cell (SPEC requirement), aggregated to mean ± std.
- **Noise model stays the existing global observation-noise wrapper** (`ObservationNoiseWrapper`, per-element iid corruption, prob 0.10). Do not build a localized noisy-TV — RQ4 covers noise beyond the noisy-TV.
- **β is the intrinsic reward scale.** For RND it is `reward_scale` (was `RND_REWARD_SCALE=0.05`); the LPM wrapper takes the same `reward_scale` knob so β is comparable. Sweep β **per method** — never tune on RND and reuse for LPM (the maze reward-scale mismatch is documented in `LPM_exploration/UPSTREAM.md`).
- **LPM must be paper-faithful log-space** (Eq 1–3 + Alg 1 gating), mirroring `LPM_exploration/Miniworld/experiments/models.py:LPMModel(reward_space="log")`.
- **No LaTeX in any docs/markdown** — inline linear notation only (e.g. `r = g_phi - log(MSE)`), per `CLAUDE.md`.
- This is OUR code (a rebuild), not the `LPM_exploration/` upstream snapshot, so no `UPSTREAM.md` entry is required for it.
- The source clone is at `/tmp/minigrid_intrinsic_reward` (re-clone with `gh repo clone JosefGh/minigrid_intrinsic_reward` if absent).

---

### Task 1: Vendor the project + venv deps + smoke-verify it trains

**Files:**
- Create: `minigrid_exp/` (copied from the source clone: `config.py`, `algorithms.py`, `ucb_dqn.py`, `train.py`, `run_experiments.py`, `evaluate.py`, `plot_results.py`, `record_agent.py`, `wrappers/env_factory.py`, `wrappers/noise_wrapper.py`, `wrappers/rnd_wrapper.py`, `wrappers/__init__.py`)
- Create: `minigrid_exp/__init__.py` (empty)
- Create: `minigrid_exp/README.md` (provenance note)
- Modify: `.gitignore` (ignore `minigrid_exp/results/`, `minigrid_exp/plots/` defensive backstops)

**Interfaces:**
- Consumes: nothing (entry task).
- Produces: an importable `minigrid_exp` package whose `wrappers.env_factory.make_env(env_id, intrinsic, noise, seed, training)` returns a working env, and a venv with `minigrid` + `stable-baselines3` installed.

- [ ] **Step 1: Copy the source files into `minigrid_exp/`**

```bash
cd /data/yingte/projects/ChallengingRL-LPM
test -d /tmp/minigrid_intrinsic_reward || gh repo clone JosefGh/minigrid_intrinsic_reward /tmp/minigrid_intrinsic_reward
mkdir -p minigrid_exp/wrappers
cp /tmp/minigrid_intrinsic_reward/{config.py,algorithms.py,ucb_dqn.py,train.py,run_experiments.py,evaluate.py,plot_results.py,record_agent.py} minigrid_exp/
cp /tmp/minigrid_intrinsic_reward/wrappers/{env_factory.py,noise_wrapper.py,rnd_wrapper.py} minigrid_exp/wrappers/
touch minigrid_exp/__init__.py minigrid_exp/wrappers/__init__.py
```

- [ ] **Step 2: Add a provenance README**

Create `minigrid_exp/README.md`:

```markdown
# minigrid_exp — MiniGrid intrinsic-reward experiments

Incorporated + rebuilt from the private repo `github.com/JosefGh/minigrid_intrinsic_reward`
(Youssef), 2026-06-17. SB3 DQN(UCB)/PPO on flat MiniGrid observations, with a
global observation-noise wrapper and an RND intrinsic-reward wrapper. This repo's
additions: a paper-faithful LPM wrapper (`wrappers/lpm_wrapper.py`), explicit
`method`/`beta` parameters, a process-parallel grid runner (`run_grid.py`), and
analysis writing to `expr_data/minigrid/`. See `docs/minigrid_setup_analysis.md`
and `docs/SPEC.md`.
```

- [ ] **Step 3: Install the deps into the existing venv**

```bash
cd /data/yingte/projects/ChallengingRL-LPM
./LPM_exploration/.venv/bin/python -m pip install "stable-baselines3" "minigrid" "tensorboard" "imageio"
```

- [ ] **Step 4: Verify imports + env construction + a 200-step train smoke**

Run:

```bash
cd /data/yingte/projects/ChallengingRL-LPM/minigrid_exp
PYTHONPATH=. ../LPM_exploration/.venv/bin/python -c "
from wrappers.env_factory import make_env
from algorithms import get_algorithm_class
env = make_env('MiniGrid-FourRooms-v0', intrinsic=False, noise=True, seed=0, training=True)
obs, info = env.reset(seed=0)
print('obs shape', obs.shape, 'n_actions', env.action_space.n)
m = get_algorithm_class()(policy='MlpPolicy', env=env, seed=0,
    exploration_fraction=0.0, exploration_initial_eps=0.0, exploration_final_eps=0.0,
    ucb_coefficient=1.0, ucb_state_round_decimals=None, learning_starts=10, buffer_size=1000)
m.learn(total_timesteps=200)
print('SMOKE OK')
"
```

Expected: prints `obs shape ...`, `n_actions 5`, then `SMOKE OK`. If SB3 raises a
gymnasium-API error (e.g. `reset()`/`step()` signature mismatch), the installed
gymnasium (1.3.0) is newer than this SB3 supports — fix by `pip install -U
"stable-baselines3>=2.4"` (gymnasium-1.x compatible) and re-run; record the working
SB3 version in `minigrid_exp/README.md`.

- [ ] **Step 5: Add defensive gitignore + commit**

Add to `.gitignore`:

```
# minigrid_exp local artifacts (live data is under /expr_data/minigrid/)
minigrid_exp/results/
minigrid_exp/plots/
```

```bash
cd /data/yingte/projects/ChallengingRL-LPM
git add minigrid_exp .gitignore
git commit -m "minigrid_exp: vendor Youssef's MiniGrid intrinsic-reward project"
```

---

### Task 2: Repoint artifacts to expr_data/minigrid + 8 seeds + enable the SPEC matrix

**Files:**
- Modify: `minigrid_exp/config.py` (RESULTS/PLOTS dirs → `expr_data/minigrid`; `SEEDS`; `ENVIRONMENTS`; `VARIANTS`)

**Interfaces:**
- Consumes: the vendored `config.py` from Task 1.
- Produces: `config.SEEDS == [1..8]`; `config.ENVIRONMENTS` with easy/medium/hard tiers; `config.VARIANTS` with all four cells active; `config.RESULTS_DIR` pointing under the repo's `expr_data/minigrid`.

- [ ] **Step 1: Repoint the artifact roots to expr_data/minigrid**

In `minigrid_exp/config.py`, replace the `RESULTS_DIR`/`PLOTS_DIR` block (the lines defining `BASE_DIR`, `RESULTS_DIR`, `PLOTS_DIR`) with:

```python
BASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = BASE_DIR.parent
EXPR_DATA = REPO_ROOT / "expr_data" / "minigrid"

RESULTS_DIR = EXPR_DATA / "results"
```

and change the `PLOTS_DIR` line to:

```python
PLOTS_DIR = EXPR_DATA / "figures" / ALGORITHM_NAME
```

(Leave `LOGS_DIR`, `MODELS_DIR`, `VIDEOS_DIR`, `EVAL_*_PATH` as-is — they already derive from `RESULTS_DIR`, so they follow automatically. The `*.mkdir(...)` calls below them stay.)

- [ ] **Step 2: Set 8 seeds and the full difficulty ladder**

In `minigrid_exp/config.py`, replace the `ENVIRONMENTS = {...}` and `SEEDS = [...]` blocks with:

```python
ENVIRONMENTS = {
    "easy": [
        "MiniGrid-Empty-8x8-v0",
        "MiniGrid-DoorKey-5x5-v0",
    ],
    "medium": [
        "MiniGrid-FourRooms-v0",
        "MiniGrid-DoorKey-8x8-v0",
    ],
    "hard": [
        "MiniGrid-MultiRoom-N6-v0",
        "MiniGrid-KeyCorridorS3R3-v0",
    ],
}

SEEDS = [1, 2, 3, 4, 5, 6, 7, 8]
```

- [ ] **Step 3: Enable all four variants**

In `minigrid_exp/config.py`, replace the entire `VARIANTS = [...]` definition (including the commented triple-quoted blocks after it) with:

```python
VARIANTS = [
    {"name": "baseline_no_noise",  "intrinsic": False, "noise": False},
    {"name": "intrinsic_no_noise", "intrinsic": True,  "noise": False},
    {"name": "baseline_noise",     "intrinsic": False, "noise": True},
    {"name": "intrinsic_noise",    "intrinsic": True,  "noise": True},
]
```

- [ ] **Step 4: Verify config loads and points at expr_data**

Run:

```bash
cd /data/yingte/projects/ChallengingRL-LPM/minigrid_exp
PYTHONPATH=. ../LPM_exploration/.venv/bin/python -c "
import config
print('SEEDS', config.SEEDS)
print('n_variants', len(config.VARIANTS))
print('envs', sum(len(v) for v in config.ENVIRONMENTS.values()))
assert str(config.RESULTS_DIR).endswith('expr_data/minigrid/results'), config.RESULTS_DIR
assert len(config.SEEDS) == 8
assert len(config.VARIANTS) == 4
print('CONFIG OK', config.RESULTS_DIR)
"
```

Expected: `SEEDS [1, 2, 3, 4, 5, 6, 7, 8]`, `n_variants 4`, `envs 6`, then `CONFIG OK .../expr_data/minigrid/results`.

- [ ] **Step 5: Commit**

```bash
cd /data/yingte/projects/ChallengingRL-LPM
git add minigrid_exp/config.py
git commit -m "minigrid_exp: artifacts -> expr_data/minigrid, 8 seeds, full env+variant matrix"
```

---

### Task 3: Add the paper-faithful LPM intrinsic-reward wrapper

**Files:**
- Create: `minigrid_exp/wrappers/lpm_wrapper.py`
- Modify: `minigrid_exp/wrappers/env_factory.py` (accept `method`/`beta`, dispatch RND vs LPM)
- Modify: `minigrid_exp/config.py` (add `LPM_*` knobs; `INTRINSIC_REWARD_METHOD` allows `"lpm"`)
- Test: `minigrid_exp/tests/test_lpm_wrapper.py`

**Interfaces:**
- Consumes: `wrappers.rnd_wrapper.RunningMeanStd`; a gymnasium env with `Discrete` actions and a flat (Box) observation.
- Produces:
  - `LPMIntrinsicRewardWrapper(env, reward_scale=0.05, learning_rate=1e-3, hidden_dim=128, buffer_size=100, normalize_observations=True, normalize_rewards=True, observation_clip=5.0, device="auto", seed=None)` — a `gym.Wrapper`; `step()` returns `(obs, total_reward, terminated, truncated, info)` with `info["extrinsic_reward"]`, `info["lpm_raw_intrinsic_reward"]`, `info["lpm_intrinsic_reward"]`. Intrinsic reward is exactly 0.0 until `buffer_size` transitions have been seen (Alg 1 L6 gating).
  - `make_env(env_id, intrinsic=False, noise=False, seed=0, noise_prob=0.10, training=False, method="rnd", beta=None)` — `method` in `{"rnd","lpm"}`; `beta`, if not None, overrides the wrapper's `reward_scale`.

- [ ] **Step 1: Write the failing test**

Create `minigrid_exp/tests/test_lpm_wrapper.py`:

```python
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from wrappers.env_factory import make_env
from wrappers.lpm_wrapper import LPMIntrinsicRewardWrapper


def _rollout(env, n, seed=0):
    env.reset(seed=seed)
    rows = []
    for _ in range(n):
        obs, r, term, trunc, info = env.step(env.action_space.sample())
        rows.append(info)
        if term or trunc:
            env.reset()
    return rows


def test_info_has_extrinsic_intrinsic_split():
    env = make_env("MiniGrid-Empty-8x8-v0", intrinsic=True, noise=False,
                   seed=0, training=True, method="lpm", beta=0.05)
    inner = env
    while not isinstance(inner, LPMIntrinsicRewardWrapper):
        inner = inner.env
    rows = _rollout(env, 5)
    assert all("lpm_intrinsic_reward" in r and "extrinsic_reward" in r for r in rows)


def test_reward_gated_to_zero_before_buffer_fills():
    # buffer_size=10: the first <10 steps must yield exactly 0 intrinsic reward.
    base = make_env("MiniGrid-Empty-8x8-v0", intrinsic=False, noise=False,
                    seed=0, training=True)
    env = LPMIntrinsicRewardWrapper(base, buffer_size=10, reward_scale=1.0, seed=0)
    rows = _rollout(env, 8)
    assert all(r["lpm_intrinsic_reward"] == 0.0 for r in rows), \
        [r["lpm_intrinsic_reward"] for r in rows]


def test_reward_nonzero_after_buffer_fills():
    base = make_env("MiniGrid-Empty-8x8-v0", intrinsic=False, noise=False,
                    seed=1, training=True)
    env = LPMIntrinsicRewardWrapper(base, buffer_size=5, reward_scale=1.0, seed=1)
    rows = _rollout(env, 60)
    later = [r["lpm_intrinsic_reward"] for r in rows[20:]]
    assert any(abs(x) > 0.0 for x in later), later
```

- [ ] **Step 2: Run the test to verify it fails**

Run:

```bash
cd /data/yingte/projects/ChallengingRL-LPM/minigrid_exp
PYTHONPATH=. ../LPM_exploration/.venv/bin/python -m pytest tests/test_lpm_wrapper.py -q
```

Expected: FAIL with `ModuleNotFoundError: No module named 'wrappers.lpm_wrapper'` (and `make_env` not yet accepting `method`).

- [ ] **Step 3: Implement the LPM wrapper**

Create `minigrid_exp/wrappers/lpm_wrapper.py`:

```python
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
    scaled by reward_scale, mirroring RNDIntrinsicRewardWrapper so beta is
    comparable across methods. Faithful to
    LPM_exploration/Miniworld/experiments/models.py:LPMModel(reward_space="log").
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
            raw = 0.0
        else:
            raw = float(g - float(np.log(mse + 1e-6)))

        if self.normalize_rewards:
            self.reward_rms.update(np.asarray([raw], dtype=np.float64))
            bonus = raw / np.sqrt(self.reward_rms.var + 1e-8)
        else:
            bonus = raw
        return raw, float(self.reward_scale * bonus)
```

- [ ] **Step 4: Wire `method`/`beta` into `make_env`**

In `minigrid_exp/wrappers/env_factory.py`: add the import near the other wrapper imports:

```python
from wrappers.lpm_wrapper import LPMIntrinsicRewardWrapper
```

Change the `make_env` signature line from:

```python
def make_env(
    env_id: str,
    intrinsic: bool = False,
    noise: bool = False,
    seed: int = 0,
    noise_prob: float = 0.10,
    training: bool = False,
):
```

to add `method` and `beta`:

```python
def make_env(
    env_id: str,
    intrinsic: bool = False,
    noise: bool = False,
    seed: int = 0,
    noise_prob: float = 0.10,
    training: bool = False,
    method: str = "rnd",
    beta: float | None = None,
):
```

Replace the existing `if training and intrinsic:` block (the one that raises for non-rnd and constructs `RNDIntrinsicRewardWrapper`) with:

```python
    if training and intrinsic:
        if method == "rnd":
            env = RNDIntrinsicRewardWrapper(
                env,
                reward_scale=RND_REWARD_SCALE if beta is None else beta,
                learning_rate=RND_LEARNING_RATE,
                hidden_dim=RND_HIDDEN_DIM,
                output_dim=RND_OUTPUT_DIM,
                normalize_observations=RND_NORMALIZE_OBSERVATIONS,
                normalize_rewards=RND_NORMALIZE_REWARDS,
                observation_clip=RND_OBSERVATION_CLIP,
                device=RND_DEVICE,
                seed=seed,
            )
        elif method == "lpm":
            env = LPMIntrinsicRewardWrapper(
                env,
                reward_scale=LPM_REWARD_SCALE if beta is None else beta,
                learning_rate=LPM_LEARNING_RATE,
                hidden_dim=LPM_HIDDEN_DIM,
                buffer_size=LPM_BUFFER_SIZE,
                normalize_observations=RND_NORMALIZE_OBSERVATIONS,
                normalize_rewards=RND_NORMALIZE_REWARDS,
                observation_clip=RND_OBSERVATION_CLIP,
                device=RND_DEVICE,
                seed=seed,
            )
        else:
            raise ValueError(f"Unsupported intrinsic reward method: {method}")
```

Update the config import block at the top of `env_factory.py` to also import the LPM knobs — add these names to the existing `from config import (...)` list:

```python
    LPM_REWARD_SCALE,
    LPM_LEARNING_RATE,
    LPM_HIDDEN_DIM,
    LPM_BUFFER_SIZE,
```

(`INTRINSIC_REWARD_METHOD` is no longer read inside `make_env` — selection is now via the `method` parameter — so it may stay imported or be removed; leaving it imported is harmless.)

- [ ] **Step 5: Add LPM config knobs**

In `minigrid_exp/config.py`, immediately after the `RND_*` block (after `RND_DEVICE = "auto"`), add:

```python
LPM_REWARD_SCALE = 0.05
LPM_LEARNING_RATE = 1e-3
LPM_HIDDEN_DIM = 128
LPM_BUFFER_SIZE = 100
```

- [ ] **Step 6: Run the tests to verify they pass**

Run:

```bash
cd /data/yingte/projects/ChallengingRL-LPM/minigrid_exp
PYTHONPATH=. ../LPM_exploration/.venv/bin/python -m pytest tests/test_lpm_wrapper.py -q
```

Expected: 3 passed.

- [ ] **Step 7: Commit**

```bash
cd /data/yingte/projects/ChallengingRL-LPM
git add minigrid_exp/wrappers/lpm_wrapper.py minigrid_exp/wrappers/env_factory.py minigrid_exp/config.py minigrid_exp/tests/test_lpm_wrapper.py
git commit -m "minigrid_exp: add paper-faithful log-space LPM intrinsic-reward wrapper"
```

---

### Task 4: Parametrized single-run CLI + process-parallel grid runner with β sweep

**Files:**
- Modify: `minigrid_exp/train.py` (give `train_agent` explicit `method`, `beta`, `tag` params; thread them into `make_env`)
- Create: `minigrid_exp/train_one.py` (CLI around `train_agent`)
- Create: `minigrid_exp/run_grid.py` (process-parallel launcher)

**Interfaces:**
- Consumes: `train.train_agent`; `config` (SEEDS, ENVIRONMENTS, TOTAL_TIMESTEPS, LOGS_DIR, MODELS_DIR).
- Produces:
  - `train_agent(env_id, variant_name, intrinsic, noise, seed, total_timesteps, log_dir, model_dir, method="rnd", beta=None, tag=None)` — `run_name` includes `method`/`beta`/`tag` so cells don't collide.
  - `train_one.py` CLI flags: `--env --intrinsic --noise --method {rnd,lpm,none} --beta FLOAT --seed INT --steps INT`.
  - `run_grid.py` CLI flags: `--steps --seeds --jobs --betas --dry-run`, completion-marked by the saved model `.zip`.

- [ ] **Step 1: Thread `method`/`beta`/`tag` through `train_agent`**

In `minigrid_exp/train.py`, change the `train_agent` signature to add the three params (default-valued, so existing callers keep working):

```python
def train_agent(
    env_id: str,
    variant_name: str,
    intrinsic: bool,
    noise: bool,
    seed: int,
    total_timesteps: int,
    log_dir,
    model_dir,
    method: str = "rnd",
    beta: float | None = None,
    tag: str | None = None,
):
```

Change the `run_name` line from:

```python
    run_name = f"{env_id}__{variant_name}__seed_{seed}"
```

to:

```python
    suffix = f"__{tag}" if tag else ""
    run_name = f"{env_id}__{variant_name}__{method}__seed_{seed}{suffix}"
```

In the same function, both `make_env(...)` calls (the `n_envs == 1` training env and the `eval_env`) and the `make_vector_env(...)` call must forward `method`/`beta`. For the two direct `make_env(...)` calls add `method=method, beta=beta` to their kwargs. For `make_vector_env`, add `method`/`beta` params to its signature and forward them into the inner `make_env`:

```python
def make_vector_env(env_id, intrinsic, noise, seed, training, n_envs, log_dir, run_name,
                    method="rnd", beta=None):
    ...
        def _make_env(env_seed=env_seed):
            return make_env(
                env_id=env_id,
                intrinsic=intrinsic,
                noise=noise,
                seed=env_seed,
                training=training,
                method=method,
                beta=beta,
            )
```

and at its call site pass `method=method, beta=beta`.

- [ ] **Step 2: Create the single-run CLI**

Create `minigrid_exp/train_one.py`:

```python
"""Train one MiniGrid cell from the CLI (one process per run).

Example:
  PYTHONPATH=. python train_one.py --env MiniGrid-FourRooms-v0 \
      --intrinsic --noise --method lpm --beta 0.05 --seed 1 --steps 1000000
"""
import argparse

from config import LOGS_DIR, MODELS_DIR, TOTAL_TIMESTEPS
from train import train_agent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", required=True)
    ap.add_argument("--intrinsic", action="store_true")
    ap.add_argument("--noise", action="store_true")
    ap.add_argument("--method", default="rnd", choices=["rnd", "lpm", "none"])
    ap.add_argument("--beta", type=float, default=None)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--steps", type=int, default=TOTAL_TIMESTEPS)
    a = ap.parse_args()

    intrinsic = a.intrinsic and a.method != "none"
    variant = f"{'intrinsic' if intrinsic else 'baseline'}_{'noise' if a.noise else 'no_noise'}"
    tag = None if a.beta is None else f"beta{a.beta:g}"

    train_agent(
        env_id=a.env, variant_name=variant, intrinsic=intrinsic, noise=a.noise,
        seed=a.seed, total_timesteps=a.steps, log_dir=LOGS_DIR, model_dir=MODELS_DIR,
        method=a.method, beta=a.beta, tag=tag,
    )


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Create the process-parallel grid runner**

Create `minigrid_exp/run_grid.py`:

```python
"""Process-parallel grid runner for the MiniGrid intrinsic-reward experiments.

One subprocess per (env, variant, method, beta, seed) cell — process-level
parallelism (the GIL makes thread pools poor for SB3 stepping). Resumable: a
cell is complete iff its saved model .zip exists under expr_data/minigrid.

Usage:
  PYTHONPATH=. python run_grid.py --steps 1000000 --jobs 32
  PYTHONPATH=. python run_grid.py --betas 0.0 0.01 0.05 0.1 0.5 --jobs 32   # beta sweep
  PYTHONPATH=. python run_grid.py --dry-run
"""
import argparse
import itertools
import os
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed

import config

EXP = os.path.dirname(os.path.abspath(__file__))
PY = os.path.join(os.path.dirname(EXP), "LPM_exploration", ".venv", "bin", "python")

# (variant_name, intrinsic, noise)
VARIANTS = [(v["name"], v["intrinsic"], v["noise"]) for v in config.VARIANTS]


def cell_complete(env_id, variant, method, seed, tag):
    suffix = f"__{tag}" if tag else ""
    run_name = f"{env_id}__{variant}__{method}__seed_{seed}{suffix}".replace("/", "_")
    return os.path.exists(os.path.join(config.MODELS_DIR, f"{run_name}.zip"))


def run_cell(cmd, logfile):
    with open(logfile, "w") as fh:
        return subprocess.run(cmd, cwd=EXP, stdout=fh, stderr=subprocess.STDOUT,
                              env={**os.environ, "PYTHONPATH": EXP,
                                   "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}).returncode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=config.TOTAL_TIMESTEPS)
    ap.add_argument("--seeds", type=int, nargs="+", default=config.SEEDS)
    ap.add_argument("--methods", nargs="+", default=["rnd", "lpm"])
    ap.add_argument("--betas", type=float, nargs="+", default=[None],
                    help="intrinsic-reward scales to sweep; None = config default")
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    os.makedirs("/tmp/minigrid_logs", exist_ok=True)
    envs = [e for tier in config.ENVIRONMENTS.values() for e in tier]

    pending = []
    for env_id, (variant, intrinsic, noise), seed in itertools.product(envs, VARIANTS, a.seeds):
        # Non-intrinsic baselines run once per method-agnostic cell; tag by "none".
        methods = a.methods if intrinsic else ["none"]
        betas = a.betas if intrinsic else [None]
        for method, beta in itertools.product(methods, betas):
            tag = None if beta is None else f"beta{beta:g}"
            if cell_complete(env_id, variant, method, seed, tag):
                continue
            cmd = [PY, os.path.join(EXP, "train_one.py"), "--env", env_id,
                   "--method", method, "--seed", str(seed), "--steps", str(a.steps)]
            if intrinsic:
                cmd.append("--intrinsic")
            if noise:
                cmd.append("--noise")
            if beta is not None:
                cmd += ["--beta", str(beta)]
            rid = f"{env_id}__{variant}__{method}__s{seed}" + (f"__{tag}" if tag else "")
            pending.append((rid.replace("/", "_"), cmd))

    print(f"{len(pending)} cells to run, jobs={a.jobs}")
    if a.dry_run:
        for rid, _ in pending:
            print("[run]", rid)
        return

    with ProcessPoolExecutor(max_workers=a.jobs) as ex:
        futs = {ex.submit(run_cell, cmd, f"/tmp/minigrid_logs/{rid}.out"): rid
                for rid, cmd in pending}
        for fut in as_completed(futs):
            rid = futs[fut]
            rc = fut.result()
            print(f"[{'done' if rc == 0 else 'FAIL'}] {rid}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Dry-run to verify the cell matrix**

Run:

```bash
cd /data/yingte/projects/ChallengingRL-LPM/minigrid_exp
PYTHONPATH=. ../LPM_exploration/.venv/bin/python run_grid.py --dry-run --seeds 1 2 | tail -20
PYTHONPATH=. ../LPM_exploration/.venv/bin/python run_grid.py --dry-run --betas 0.0 0.05 0.5 --seeds 1 | grep -c beta
```

Expected: the first prints `[run]` lines covering 6 envs × 4 variants with `none` for baselines and `rnd`/`lpm` for intrinsic cells; the second prints a non-zero count of `beta`-tagged cells.

- [ ] **Step 5: Smoke one real short cell end-to-end**

Run:

```bash
cd /data/yingte/projects/ChallengingRL-LPM/minigrid_exp
PYTHONPATH=. ../LPM_exploration/.venv/bin/python train_one.py \
  --env MiniGrid-Empty-8x8-v0 --intrinsic --method lpm --beta 0.05 --seed 1 --steps 3000
ls ../expr_data/minigrid/models/*/MiniGrid-Empty-8x8-v0__intrinsic_no_noise__lpm__seed_1__beta0.05.zip
```

Expected: training runs without error and the `.zip` exists under `expr_data/minigrid/models/<algo>/`.

- [ ] **Step 6: Commit**

```bash
cd /data/yingte/projects/ChallengingRL-LPM
git add minigrid_exp/train.py minigrid_exp/train_one.py minigrid_exp/run_grid.py
git commit -m "minigrid_exp: parametrized train_one CLI + process-parallel run_grid with beta sweep"
```

---

### Task 5: Aggregation + analysis (success rate, sample-efficiency curves, intrinsic/extrinsic split)

**Files:**
- Create: `minigrid_exp/analyze.py`
- Create: `expr_data/minigrid/README.md` (markdown explanation of the data layout — SPEC observability requirement)
- Test: `minigrid_exp/tests/test_analyze.py`

**Interfaces:**
- Consumes: SB3 `EvalCallback` outputs at `expr_data/minigrid/logs/<algo>/eval/<run_name>/evaluations.npz` (keys `timesteps`, `results` shape `(n_evals, n_episodes)`), and `Monitor` CSVs at `expr_data/minigrid/logs/<algo>/<run_name>.monitor.csv`.
- Produces:
  - `aggregate_eval_curves(logs_dir) -> pandas.DataFrame` with columns `[env, variant, method, beta, seed, timestep, mean_return]`.
  - `summarize(df) -> pandas.DataFrame` grouped by `[env, variant, method, beta, timestep]` with `mean`/`std` over seeds.
  - figures under `expr_data/minigrid/figures/`: per-env `fig_sample_efficiency_<env>.png` (return vs. timestep, mean±std band per method/variant) and `table_final_success.csv` (final-window mean return per cell).

- [ ] **Step 1: Write the failing test (parser on a synthetic evaluations.npz)**

Create `minigrid_exp/tests/test_analyze.py`:

```python
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import analyze


def test_parse_run_name():
    p = analyze.parse_run_name(
        "MiniGrid-FourRooms-v0__intrinsic_noise__lpm__seed_3__beta0.05")
    assert p == {"env": "MiniGrid-FourRooms-v0", "variant": "intrinsic_noise",
                 "method": "lpm", "seed": 3, "beta": "0.05"}


def test_parse_run_name_no_beta():
    p = analyze.parse_run_name(
        "MiniGrid-Empty-8x8-v0__baseline_no_noise__none__seed_1")
    assert p["beta"] is None and p["method"] == "none" and p["seed"] == 1


def test_load_eval_npz(tmp_path):
    d = tmp_path / "eval" / "MiniGrid-Empty-8x8-v0__baseline_no_noise__none__seed_1"
    d.mkdir(parents=True)
    np.savez(d / "evaluations.npz",
             timesteps=np.array([100, 200]),
             results=np.array([[0.1, 0.3], [0.5, 0.7]]),
             ep_lengths=np.array([[10, 10], [9, 9]]))
    rows = analyze.load_eval_npz(str(d / "evaluations.npz"))
    assert [r["timestep"] for r in rows] == [100, 200]
    assert abs(rows[1]["mean_return"] - 0.6) < 1e-9
```

- [ ] **Step 2: Run the test to verify it fails**

Run:

```bash
cd /data/yingte/projects/ChallengingRL-LPM/minigrid_exp
PYTHONPATH=. ../LPM_exploration/.venv/bin/python -m pytest tests/test_analyze.py -q
```

Expected: FAIL with `ModuleNotFoundError: No module named 'analyze'`.

- [ ] **Step 3: Implement analyze.py**

Create `minigrid_exp/analyze.py`:

```python
"""Aggregate MiniGrid runs -> sample-efficiency curves + final-success table.

Reads SB3 EvalCallback outputs under expr_data/minigrid/logs/<algo>/eval/<run>/
and writes figures + a summary CSV under expr_data/minigrid/figures/.
"""
from __future__ import annotations

import argparse
import glob
import os
import re

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config

RUN_RE = re.compile(
    r"^(?P<env>.+?)__(?P<variant>baseline_no_noise|baseline_noise|"
    r"intrinsic_no_noise|intrinsic_noise)__(?P<method>rnd|lpm|none)__seed_(?P<seed>\d+)"
    r"(?:__beta(?P<beta>[0-9.eE+-]+))?$")


def parse_run_name(name: str):
    m = RUN_RE.match(name)
    if not m:
        return None
    d = m.groupdict()
    return {"env": d["env"], "variant": d["variant"], "method": d["method"],
            "seed": int(d["seed"]), "beta": d["beta"]}


def load_eval_npz(path: str):
    data = np.load(path)
    ts = data["timesteps"]
    res = data["results"]  # (n_evals, n_episodes)
    return [{"timestep": int(t), "mean_return": float(r.mean())}
            for t, r in zip(ts, res)]


def aggregate_eval_curves(logs_dir: str) -> pd.DataFrame:
    rows = []
    for npz in glob.glob(os.path.join(logs_dir, "eval", "*", "evaluations.npz")):
        run = os.path.basename(os.path.dirname(npz))
        meta = parse_run_name(run)
        if meta is None:
            continue
        for pt in load_eval_npz(npz):
            rows.append({**meta, **pt})
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby(["env", "variant", "method", "beta", "timestep"], dropna=False)
    return g["mean_return"].agg(["mean", "std", "count"]).reset_index()


def plot_curves(summary: pd.DataFrame, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    for env in sorted(summary["env"].unique()):
        sub = summary[summary["env"] == env]
        plt.figure(figsize=(7, 5))
        for (variant, method, beta), s in sub.groupby(["variant", "method", "beta"], dropna=False):
            s = s.sort_values("timestep")
            label = f"{variant}/{method}" + (f"/b{beta}" if beta else "")
            plt.plot(s["timestep"], s["mean"], label=label)
            plt.fill_between(s["timestep"], s["mean"] - s["std"], s["mean"] + s["std"], alpha=0.15)
        plt.xlabel("training step"); plt.ylabel("eval mean return")
        plt.title(env); plt.legend(fontsize=6); plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"fig_sample_efficiency_{env.replace('/', '_')}.png"), dpi=130)
        plt.close()


def final_success_table(df: pd.DataFrame, frac: float = 0.1) -> pd.DataFrame:
    out = []
    for keys, s in df.groupby(["env", "variant", "method", "beta", "seed"], dropna=False):
        s = s.sort_values("timestep")
        k = max(1, int(len(s) * frac))
        out.append({**dict(zip(["env", "variant", "method", "beta", "seed"], keys)),
                    "final_return": s["mean_return"].tail(k).mean()})
    fdf = pd.DataFrame(out)
    return (fdf.groupby(["env", "variant", "method", "beta"], dropna=False)["final_return"]
            .agg(["mean", "std", "count"]).reset_index())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", default=str(config.LOGS_DIR))
    ap.add_argument("--figures", default=str(config.EXPR_DATA / "figures"))
    a = ap.parse_args()
    df = aggregate_eval_curves(a.logs)
    if df.empty:
        print("no eval data found under", a.logs); return
    summary = summarize(df)
    plot_curves(summary, a.figures)
    os.makedirs(a.figures, exist_ok=True)
    final_success_table(df).to_csv(os.path.join(a.figures, "table_final_success.csv"), index=False)
    print("wrote figures + table_final_success.csv to", a.figures)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the tests to verify they pass**

Run:

```bash
cd /data/yingte/projects/ChallengingRL-LPM/minigrid_exp
PYTHONPATH=. ../LPM_exploration/.venv/bin/python -m pytest tests/test_analyze.py -q
```

Expected: 3 passed.

- [ ] **Step 5: Write the expr_data observability note**

Create `expr_data/minigrid/README.md`:

```markdown
# MiniGrid experiment data

Produced by `minigrid_exp/run_grid.py` + `analyze.py`. Gitignored.

- `models/<algo>/<run_name>.zip` — final/best SB3 model per cell. The model `.zip`
  is the completion marker `run_grid.py` resumes on.
- `logs/<algo>/<run_name>.monitor.csv` — per-episode training reward/length.
- `logs/<algo>/eval/<run_name>/evaluations.npz` — periodic eval (timesteps,
  results[n_evals, n_episodes]); the sample-efficiency curve source.
- `figures/fig_sample_efficiency_<env>.png` — eval return vs. training step,
  mean ± std over 8 seeds, one line per variant/method(/beta).
- `figures/table_final_success.csv` — final-window mean return per cell.

`run_name` = `<env>__<variant>__<method>__seed_<s>[__beta<b>]`, e.g.
`MiniGrid-FourRooms-v0__intrinsic_noise__lpm__seed_3__beta0.05`. The
`intrinsic_reward`/`extrinsic_reward` split is in each step's `info` dict
(`{rnd,lpm}_intrinsic_reward`, `extrinsic_reward`) for separate tracking.
```

- [ ] **Step 6: Commit**

```bash
cd /data/yingte/projects/ChallengingRL-LPM
git add minigrid_exp/analyze.py minigrid_exp/tests/test_analyze.py expr_data/minigrid/README.md
git commit -m "minigrid_exp: analysis (sample-efficiency curves + final-success table) + data note"
```

---

## Self-Review

**Spec coverage (`docs/SPEC.md`):**
- RQ "correct way to mix extrinsic + intrinsic / β too small or large" → Task 4 β-sweep (`run_grid.py --betas`) + Task 5 curves; intrinsic/extrinsic split in `info` (Task 3). Covered.
- RQ "does intrinsic out-perform baselines on sparse-reward MiniGrid" → Tasks 2+4 (baseline vs intrinsic variants across the difficulty ladder) + Task 5 success table. Covered.
- RQ "does LPM generalize to noise beyond noisy-TV" → Task 3 (LPM wrapper) + global-noise variants (Task 2) + LPM-vs-RND in Task 4/5. Covered.
- Work-item "pick env / understand noise" → done in `docs/minigrid_setup_analysis.md`; ladder enabled in Task 2.
- Work-item "β sweep on RND, evaluate extrinsic/intrinsic separately" → Task 4 (`--betas`), Task 3 (`info` split). Covered.
- Work-item "different methods on clean+noisy, difficulty up → intrinsic wins" → Tasks 2/4/5. Covered.
- Work-item "LPM vs RND under different noise ratio" → supported; note: the SPEC's "different ratio of observation noise" needs a `noise_prob` sweep — `make_env` already takes `noise_prob`; add `--noise-prob` to `train_one.py`/`run_grid.py` as a follow-up if a ratio sweep is wanted (not built here; flagged so it isn't silently assumed done).
- Requirements: 8 seeds (Task 2), expr_data + markdown (Tasks 2/5), parallel compute (Task 4). Covered.

**Placeholder scan:** no TBD/TODO/"handle edge cases"/"similar to Task N" — all code is given in full; vendored files are copied verbatim via explicit `cp`. OK.

**Type consistency:** `make_env(..., method, beta)` defined in Task 3 and consumed in Task 4; `train_agent(..., method, beta, tag)` defined and called consistently; `run_name` format `<env>__<variant>__<method>__seed_<s>[__beta<b>]` is identical across `train.py` (Task 4 Step 1), `run_grid.cell_complete` (Task 4 Step 3), and `analyze.RUN_RE` (Task 5 Step 3). `config.EXPR_DATA` defined in Task 2 Step 1 and used in `analyze.main` (Task 5). OK.

**Known open item (not a gap):** the observation-noise *ratio* sweep for the last work-item is parameter-ready (`noise_prob`) but not wired to the CLI here — surface to the user before that specific experiment rather than assume it's covered.
