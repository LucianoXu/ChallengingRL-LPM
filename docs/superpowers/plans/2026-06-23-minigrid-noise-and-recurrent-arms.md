# MiniGrid Noise-Model Fix + Recurrent-Policy Arms Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the per-element observation-noise model with a cell-level, range-respecting one, and add two recurrent-policy exploration arms (`rnd_lstm`, `lpm_lstm`) that train RND/LPM with a `RecurrentPPO` LSTM policy alongside `none`/`entropy`/`rnd`/`lpm`.

**Architecture:** Two independent changes to `minigrid_exp/`. (1) `ObservationNoiseWrapper` is rewritten so `noise_prob` is the per-cell Bernoulli probability and each corrupted cell is re-drawn per-channel within MiniGrid's valid encoding ranges. (2) A small `method_utils` module centralizes method-string parsing so a `_lstm` suffix selects `sb3_contrib.RecurrentPPO` + `MlpLstmPolicy` while the intrinsic wrapper is chosen from the base method; the suffix is threaded through `algorithms`/`train`/`env_factory`/`train_one`/`analyze`/`make_report_figs`.

**Tech Stack:** Python 3.11 (uv venv at `LPM_exploration/.venv`), `stable_baselines3==2.9.0`, `sb3-contrib==2.9.0` (new), `gymnasium`, `minigrid==3.1.0`, `numpy`, `torch`, `pandas`, `matplotlib`, `pytest`.

## Global Constraints

- Math/notation in any prose or comments uses inline linear notation, never LaTeX delimiters (project rule).
- Keep experiment artifacts under `expr_data/minigrid/` (gitignored).
- Venv python: `/data/yingte/projects/ChallengingRL-LPM/LPM_exploration/.venv/bin/python`.
- Tests run from `minigrid_exp/` with `PYTHONPATH=.`; test files prepend the `minigrid_exp` dir to `sys.path` (see existing tests) and import modules directly (`import analyze`, `from wrappers.env_factory import make_env`).
- MiniGrid symbolic encoding ranges (inclusive): object [0,10], color [0,5], state [0,2]. Read them from `minigrid.core.constants` (`OBJECT_TO_IDX`, `COLOR_TO_IDX`, `STATE_TO_IDX`) — do not hardcode.
- `noise_prob` default stays 0.10 (now meaning "10% of cells"); the `--noise-probs` sweep is deferred — this round runs only `0.1`.
- Recurrent LSTM config is fixed: `MlpLstmPolicy`, `lstm_hidden_size=128`, `n_lstm_layers=1`, `shared_lstm=True`, `enable_critic_lstm=False`. Recurrent arms reuse the MLP β (RND 0.005, LPM 0.001).
- Seeds: DoorKey-5x5 = 8 (1–8); FourRooms, MultiRoom-N6 = 3 (1–3). Step budgets: DoorKey 1,000,000; FourRooms 1,000,000; MultiRoom-N6 2,000,000.

---

## Task 1: Cell-level, range-respecting noise wrapper

**Files:**
- Rewrite: `minigrid_exp/wrappers/noise_wrapper.py`
- Modify: `minigrid_exp/wrappers/env_factory.py:146` (pass `seed` to the noise wrapper)
- Test: `minigrid_exp/tests/test_noise_wrapper.py`

**Interfaces:**
- Produces: `ObservationNoiseWrapper(env, noise_prob: float = 0.10, seed: int | None = None)` — a `gym.ObservationWrapper`. Exposes `last_cell_mask: np.ndarray | None` (the most recent `(H, W)` boolean cell mask) for testing/observability.

- [ ] **Step 1: Write the failing tests**

Create `minigrid_exp/tests/test_noise_wrapper.py`:

```python
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import gymnasium as gym
import minigrid  # noqa: F401  (registers MiniGrid env ids)
from minigrid.core.constants import OBJECT_TO_IDX, COLOR_TO_IDX, STATE_TO_IDX
from wrappers.noise_wrapper import ObservationNoiseWrapper


def test_zero_prob_is_noop():
    clean, _ = gym.make("MiniGrid-Empty-8x8-v0").reset(seed=0)
    wrapped = ObservationNoiseWrapper(
        gym.make("MiniGrid-Empty-8x8-v0"), noise_prob=0.0, seed=0)
    noisy, _ = wrapped.reset(seed=0)
    assert np.array_equal(clean["image"], noisy["image"])


def test_channels_stay_in_range():
    # noise_prob=1.0 corrupts every cell; assert no channel exceeds its valid max.
    env = ObservationNoiseWrapper(
        gym.make("MiniGrid-Empty-8x8-v0"), noise_prob=1.0, seed=1)
    env.reset(seed=1)
    omax, cmax, smax = (max(OBJECT_TO_IDX.values()),
                        max(COLOR_TO_IDX.values()),
                        max(STATE_TO_IDX.values()))
    for _ in range(50):
        obs, *_ = env.step(2)  # 2 = forward; deterministic given the seed
        img = obs["image"]
        assert img[..., 0].max() <= omax
        assert img[..., 1].max() <= cmax
        assert img[..., 2].max() <= smax


def test_mask_is_cell_level_and_fraction_matches():
    env = ObservationNoiseWrapper(
        gym.make("MiniGrid-Empty-8x8-v0"), noise_prob=0.3, seed=3)
    obs, _ = env.reset(seed=3)
    fracs = []
    for _ in range(200):
        obs, *_ = env.step(2)
        m = env.last_cell_mask
        assert m.shape == obs["image"].shape[:2]   # (H, W) — cell-level, not per-channel
        fracs.append(m.mean())
    assert abs(float(np.mean(fracs)) - 0.3) < 0.05


def test_same_seed_reproducible():
    def run():
        e = ObservationNoiseWrapper(
            gym.make("MiniGrid-Empty-8x8-v0"), noise_prob=0.5, seed=7)
        e.reset(seed=7)
        return [e.step(2)[0]["image"].copy() for _ in range(10)]
    a, b = run(), run()
    assert all(np.array_equal(x, y) for x, y in zip(a, b))
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /data/yingte/projects/ChallengingRL-LPM/minigrid_exp && PYTHONPATH=. ../LPM_exploration/.venv/bin/python -m pytest tests/test_noise_wrapper.py -v`
Expected: FAIL — current wrapper has no `seed` kwarg / no `last_cell_mask`; `test_channels_stay_in_range` fails because the old code uses `randint(0,10)` (color/state out of range).

- [ ] **Step 3: Rewrite the noise wrapper**

Replace the entire contents of `minigrid_exp/wrappers/noise_wrapper.py`:

```python
import gymnasium as gym
import numpy as np
from minigrid.core.constants import OBJECT_TO_IDX, COLOR_TO_IDX, STATE_TO_IDX


class ObservationNoiseWrapper(gym.ObservationWrapper):
    """Corrupt a fraction of *cells* in the MiniGrid symbolic image.

    `noise_prob` is the per-cell Bernoulli probability that a grid cell (a
    location in the H x W egocentric view) is corrupted — i.e. the expected
    fraction of cells perturbed, NOT a per-channel/per-element probability. A
    corrupted cell is re-drawn as a unit: each of its 3 channels (object, color,
    state) gets an independent uniform draw within that channel's valid MiniGrid
    range (object in [0,10], color in [0,5], state in [0,2]), so the noise never
    injects encodings the agent could not otherwise see.

    Owns a seeded numpy Generator for reproducibility. `last_cell_mask` exposes
    the most recent (H, W) boolean cell mask for testing/observability.
    """

    def __init__(self, env, noise_prob: float = 0.10, seed: int | None = None):
        super().__init__(env)
        self.noise_prob = noise_prob
        self._rng = np.random.default_rng(seed)
        # Per-channel inclusive max for the (object, color, state) encoding.
        self._channel_max = np.array(
            [max(OBJECT_TO_IDX.values()),
             max(COLOR_TO_IDX.values()),
             max(STATE_TO_IDX.values())],
            dtype=np.int64,
        )
        self.last_cell_mask = None

    def observation(self, obs):
        obs = obs.copy()
        image = obs["image"]
        h, w = image.shape[:2]

        cell_mask = self._rng.random((h, w)) < self.noise_prob
        self.last_cell_mask = cell_mask
        n = int(cell_mask.sum())
        if n:
            image = image.copy()  # only allocate when we actually corrupt
            for c in range(image.shape[2]):
                image[cell_mask, c] = self._rng.integers(
                    0, int(self._channel_max[c]) + 1, size=n, dtype=image.dtype)
            obs["image"] = image
        return obs
```

- [ ] **Step 4: Pass the env seed to the noise wrapper**

In `minigrid_exp/wrappers/env_factory.py`, change line 146 from:

```python
        env = ObservationNoiseWrapper(env, noise_prob=noise_prob)
```

to:

```python
        env = ObservationNoiseWrapper(env, noise_prob=noise_prob, seed=seed)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `cd /data/yingte/projects/ChallengingRL-LPM/minigrid_exp && PYTHONPATH=. ../LPM_exploration/.venv/bin/python -m pytest tests/test_noise_wrapper.py -v`
Expected: PASS (4 passed).

- [ ] **Step 6: Commit**

```bash
cd /data/yingte/projects/ChallengingRL-LPM
git add minigrid_exp/wrappers/noise_wrapper.py minigrid_exp/wrappers/env_factory.py minigrid_exp/tests/test_noise_wrapper.py
git commit -m "feat(minigrid): cell-level range-respecting observation noise

noise_prob is now the per-cell Bernoulli probability; corrupted cells are
re-drawn per-channel within MiniGrid's valid ranges (object<=10, color<=5,
state<=2). Seeded Generator for reproducibility; last_cell_mask exposed.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 2: Method-parsing helpers (`method_utils.py`)

**Files:**
- Create: `minigrid_exp/method_utils.py`
- Test: `minigrid_exp/tests/test_method_utils.py`

**Interfaces:**
- Produces:
  - `INTRINSIC_BASES = ("rnd", "lpm")` — base methods that add an intrinsic reward in the training path.
  - `base_intrinsic(method: str) -> str` — strips a trailing `_lstm` (`"rnd_lstm" -> "rnd"`, `"lpm" -> "lpm"`, `"none" -> "none"`).
  - `is_recurrent(method: str) -> bool` — `method.endswith("_lstm")`.
  - `is_intrinsic(method: str) -> bool` — `base_intrinsic(method) in INTRINSIC_BASES`.

- [ ] **Step 1: Write the failing test**

Create `minigrid_exp/tests/test_method_utils.py`:

```python
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from method_utils import base_intrinsic, is_recurrent, is_intrinsic


def test_base_intrinsic_strips_lstm():
    assert base_intrinsic("rnd_lstm") == "rnd"
    assert base_intrinsic("lpm_lstm") == "lpm"
    assert base_intrinsic("rnd") == "rnd"
    assert base_intrinsic("none") == "none"


def test_is_recurrent():
    assert is_recurrent("rnd_lstm") and is_recurrent("lpm_lstm")
    assert not is_recurrent("rnd") and not is_recurrent("none")


def test_is_intrinsic():
    assert is_intrinsic("rnd") and is_intrinsic("lpm")
    assert is_intrinsic("rnd_lstm") and is_intrinsic("lpm_lstm")
    assert not is_intrinsic("none") and not is_intrinsic("entropy")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /data/yingte/projects/ChallengingRL-LPM/minigrid_exp && PYTHONPATH=. ../LPM_exploration/.venv/bin/python -m pytest tests/test_method_utils.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'method_utils'`.

- [ ] **Step 3: Write the module**

Create `minigrid_exp/method_utils.py`:

```python
"""Method-string parsing shared across the MiniGrid pipeline.

A method string carries both the intrinsic-reward kind and the policy
architecture: a trailing '_lstm' selects a RecurrentPPO LSTM policy, while the
base ('rnd' / 'lpm') selects the intrinsic-reward wrapper.
"""

# Base methods that add an intrinsic reward in the training path. (count is
# implemented in env_factory but dormant — kept out so existing behavior, where
# train_one only treats rnd/lpm as intrinsic, is preserved.)
INTRINSIC_BASES = ("rnd", "lpm")

_LSTM_SUFFIX = "_lstm"


def base_intrinsic(method: str) -> str:
    """Strip a trailing '_lstm' policy-architecture suffix."""
    if method.endswith(_LSTM_SUFFIX):
        return method[: -len(_LSTM_SUFFIX)]
    return method


def is_recurrent(method: str) -> bool:
    """True iff the method uses a recurrent (LSTM) policy."""
    return method.endswith(_LSTM_SUFFIX)


def is_intrinsic(method: str) -> bool:
    """True iff the method adds an intrinsic reward (ignoring policy arch)."""
    return base_intrinsic(method) in INTRINSIC_BASES
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /data/yingte/projects/ChallengingRL-LPM/minigrid_exp && PYTHONPATH=. ../LPM_exploration/.venv/bin/python -m pytest tests/test_method_utils.py -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
cd /data/yingte/projects/ChallengingRL-LPM
git add minigrid_exp/method_utils.py minigrid_exp/tests/test_method_utils.py
git commit -m "feat(minigrid): method_utils — base_intrinsic/is_recurrent/is_intrinsic

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 3: Install sb3-contrib + recurrent algorithm wiring

**Files:**
- Modify: `minigrid_exp/config.py:113` (add LSTM policy + policy_kwargs)
- Modify: `minigrid_exp/algorithms.py` (whole file)
- Modify: `minigrid_exp/train.py` (imports; `get_algorithm_config`; `train_agent` call site at line 150)
- Test: `minigrid_exp/tests/test_recurrent_wiring.py`

**Interfaces:**
- Consumes: `method_utils.is_recurrent` (Task 2).
- Produces:
  - `config.PPO_LSTM_POLICY = "MlpLstmPolicy"`, `config.PPO_LSTM_POLICY_KWARGS` dict.
  - `algorithms.get_algorithm_class(method: str = "none")` — returns `RecurrentPPO` when `ALGORITHM_NAME == "ppo"` and `is_recurrent(method)`, else `PPO` (or the DQN class).
  - `train.get_algorithm_config(method: str = "none")` — its returned dict's `class`/`policy`/`policy_kwargs` reflect the recurrent choice.

- [ ] **Step 1: Install sb3-contrib into the venv**

```bash
cd /data/yingte/projects/ChallengingRL-LPM
uv pip install --python LPM_exploration/.venv/bin/python "sb3-contrib==2.9.0"
# Fallback if uv is unavailable:
# LPM_exploration/.venv/bin/python -m pip install "sb3-contrib==2.9.0"
LPM_exploration/.venv/bin/python -c "from sb3_contrib import RecurrentPPO; print('RecurrentPPO OK')"
```
Expected: `RecurrentPPO OK` (and an installed `sb3-contrib` matching `stable_baselines3==2.9.0`).

- [ ] **Step 2: Write the failing test**

Create `minigrid_exp/tests/test_recurrent_wiring.py`:

```python
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stable_baselines3 import PPO
from sb3_contrib import RecurrentPPO

import train


def test_mlp_method_uses_ppo():
    cfg = train.get_algorithm_config("rnd")
    assert cfg["class"] is PPO
    assert cfg["policy"] == "MlpPolicy"


def test_recurrent_method_uses_recurrent_ppo():
    cfg = train.get_algorithm_config("rnd_lstm")
    assert cfg["class"] is RecurrentPPO
    assert cfg["policy"] == "MlpLstmPolicy"
    assert cfg["policy_kwargs"]["shared_lstm"] is True
    assert cfg["policy_kwargs"]["enable_critic_lstm"] is False
    assert cfg["policy_kwargs"]["lstm_hidden_size"] == 128
```

- [ ] **Step 3: Run test to verify it fails**

Run: `cd /data/yingte/projects/ChallengingRL-LPM/minigrid_exp && PYTHONPATH=. ../LPM_exploration/.venv/bin/python -m pytest tests/test_recurrent_wiring.py -v`
Expected: FAIL — `get_algorithm_config` does not accept a `method` arg yet and always returns `PPO`/`MlpPolicy`.

- [ ] **Step 4: Add the LSTM config**

In `minigrid_exp/config.py`, immediately after line 113 (`PPO_POLICY_KWARGS = {}`), insert:

```python
# Recurrent (LSTM) policy for the rnd_lstm / lpm_lstm arms (sb3-contrib
# RecurrentPPO). shared_lstm=True + enable_critic_lstm=False: one LSTM feeds
# actor+critic (~halves recurrent compute vs a separate critic LSTM). hidden=128
# (obs is 147-dim; 256 is overkill).
PPO_LSTM_POLICY = "MlpLstmPolicy"
PPO_LSTM_POLICY_KWARGS = {
    "lstm_hidden_size": 128,
    "n_lstm_layers": 1,
    "shared_lstm": True,
    "enable_critic_lstm": False,
}
```

- [ ] **Step 5: Rewrite `algorithms.py`**

Replace the entire contents of `minigrid_exp/algorithms.py`:

```python
from stable_baselines3 import DQN, PPO

from config import ALGORITHM_NAME, DQN_EXPLORATION_STRATEGY
from method_utils import is_recurrent
from ucb_dqn import UCBDQN


def get_dqn_class():
    if DQN_EXPLORATION_STRATEGY == "epsilon_greedy":
        return DQN
    if DQN_EXPLORATION_STRATEGY == "ucb":
        return UCBDQN
    raise ValueError(
        f"Unsupported DQN exploration strategy: {DQN_EXPLORATION_STRATEGY}"
    )


def get_algorithm_class(method: str = "none"):
    if ALGORITHM_NAME == "ppo":
        if is_recurrent(method):
            from sb3_contrib import RecurrentPPO
            return RecurrentPPO
        return PPO
    if ALGORITHM_NAME == "dqn":
        return get_dqn_class()
    raise ValueError(f"Unsupported algorithm: {ALGORITHM_NAME}")
```

- [ ] **Step 6: Thread `method` through `get_algorithm_config` and its call site in `train.py`**

In `minigrid_exp/train.py`:

(6a) Add `PPO_LSTM_POLICY` and `PPO_LSTM_POLICY_KWARGS` to the `from config import (...)` block (alongside `PPO_POLICY`, `PPO_POLICY_KWARGS`).

(6b) Add a new import line after the `from config import (...)` block:

```python
from method_utils import is_recurrent
```

(6c) Replace the PPO branch of `get_algorithm_config` (currently the `if ALGORITHM_NAME == "ppo":` block returning the dict) so the function takes `method` and selects the recurrent policy:

```python
def get_algorithm_config(method: str = "none"):
    if ALGORITHM_NAME == "ppo":
        recurrent = is_recurrent(method)
        return {
            "class": get_algorithm_class(method),
            "policy": PPO_LSTM_POLICY if recurrent else PPO_POLICY,
            "policy_kwargs": dict(
                PPO_LSTM_POLICY_KWARGS if recurrent else PPO_POLICY_KWARGS),
            "hyperparams": dict(PPO_HYPERPARAMS),
            "eval_freq": max(PPO_EVAL_FREQ // PPO_N_ENVS, 1),
            "eval_episodes": PPO_EVAL_EPISODES,
            "n_envs": PPO_N_ENVS,
        }

    if ALGORITHM_NAME == "dqn":
```

(Leave the entire `if ALGORITHM_NAME == "dqn":` body and the trailing `raise ValueError(...)` unchanged.)

(6d) At `minigrid_exp/train.py:150`, change:

```python
    algorithm_config = get_algorithm_config()
```

to:

```python
    algorithm_config = get_algorithm_config(method)
```

- [ ] **Step 7: Run test to verify it passes**

Run: `cd /data/yingte/projects/ChallengingRL-LPM/minigrid_exp && PYTHONPATH=. ../LPM_exploration/.venv/bin/python -m pytest tests/test_recurrent_wiring.py -v`
Expected: PASS (2 passed).

- [ ] **Step 8: Commit**

```bash
cd /data/yingte/projects/ChallengingRL-LPM
git add minigrid_exp/config.py minigrid_exp/algorithms.py minigrid_exp/train.py minigrid_exp/tests/test_recurrent_wiring.py
git commit -m "feat(minigrid): RecurrentPPO wiring for *_lstm methods

method ending in _lstm selects sb3-contrib RecurrentPPO + MlpLstmPolicy
(shared LSTM, hidden 128); otherwise PPO + MlpPolicy. Adds sb3-contrib dep.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 4: env_factory dispatch + train_one choices (base-method aware)

**Files:**
- Modify: `minigrid_exp/wrappers/env_factory.py` (import; intrinsic dispatch block at lines 165–199)
- Modify: `minigrid_exp/train_one.py` (import; `--method` choices; intrinsic detection at line 27)
- Test: `minigrid_exp/tests/test_method_arms.py`

**Interfaces:**
- Consumes: `method_utils.base_intrinsic`, `method_utils.is_intrinsic` (Task 2); `RecurrentPPO` (Task 3 install).
- Produces: `make_env(..., method="rnd_lstm")` wraps with `RNDIntrinsicRewardWrapper`; `method="lpm_lstm"` wraps with `LPMIntrinsicRewardWrapper`.

- [ ] **Step 1: Write the failing tests**

Create `minigrid_exp/tests/test_method_arms.py`:

```python
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stable_baselines3.common.vec_env import DummyVecEnv
from sb3_contrib import RecurrentPPO

from wrappers.env_factory import make_env
from wrappers.rnd_wrapper import RNDIntrinsicRewardWrapper
from wrappers.lpm_wrapper import LPMIntrinsicRewardWrapper


def _has_wrapper(env, cls):
    inner = env
    while True:
        if isinstance(inner, cls):
            return True
        if not hasattr(inner, "env"):
            return False
        inner = inner.env


def test_rnd_lstm_builds_rnd_wrapper():
    env = make_env("MiniGrid-Empty-8x8-v0", intrinsic=True, training=True,
                   method="rnd_lstm", beta=0.005)
    assert _has_wrapper(env, RNDIntrinsicRewardWrapper)


def test_lpm_lstm_builds_lpm_wrapper():
    env = make_env("MiniGrid-Empty-8x8-v0", intrinsic=True, training=True,
                   method="lpm_lstm", beta=0.001)
    assert _has_wrapper(env, LPMIntrinsicRewardWrapper)


def test_recurrent_ppo_constructs_and_steps():
    # Full integration: RND wrapper (base of rnd_lstm) + RecurrentPPO +
    # MlpLstmPolicy + shared-LSTM kwargs all instantiate and train one rollout.
    env = DummyVecEnv([lambda: make_env(
        "MiniGrid-Empty-8x8-v0", intrinsic=True, training=True,
        method="rnd_lstm", beta=0.005)])
    model = RecurrentPPO(
        "MlpLstmPolicy", env, n_steps=64, batch_size=64, verbose=0,
        policy_kwargs=dict(lstm_hidden_size=64, n_lstm_layers=1,
                           shared_lstm=True, enable_critic_lstm=False))
    model.learn(total_timesteps=64)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /data/yingte/projects/ChallengingRL-LPM/minigrid_exp && PYTHONPATH=. ../LPM_exploration/.venv/bin/python -m pytest tests/test_method_arms.py -v`
Expected: FAIL — `make_env(method="rnd_lstm")` raises `ValueError: Unsupported intrinsic reward method: rnd_lstm` (dispatch keys on the exact string).

- [ ] **Step 3: Make the intrinsic dispatch base-method aware**

In `minigrid_exp/wrappers/env_factory.py`, add to the imports near line 35 (after the other `from wrappers...` imports):

```python
from method_utils import base_intrinsic
```

Then replace the intrinsic-dispatch block (lines 165–199, from `if training and intrinsic:` through the `raise ValueError(...)`) with:

```python
    if training and intrinsic:
        base = base_intrinsic(method)
        if base == "rnd":
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
        elif base == "lpm":
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
        elif base == "count":
            env = CountBasedExplorationWrapper(
                env,
                reward_scale=COUNT_REWARD_SCALE if beta is None else beta,
                seed=seed,
            )
        else:
            raise ValueError(f"Unsupported intrinsic reward method: {method}")
```

- [ ] **Step 4: Update `train_one.py` choices + intrinsic detection**

In `minigrid_exp/train_one.py`:

(4a) Add an import after `from train import train_agent`:

```python
from method_utils import is_intrinsic
```

(4b) Change the `--method` argument (line 18) from:

```python
    ap.add_argument("--method", default="rnd", choices=["rnd", "lpm", "count", "entropy", "none"])
```

to:

```python
    ap.add_argument("--method", default="rnd",
                    choices=["rnd", "lpm", "rnd_lstm", "lpm_lstm", "count", "entropy", "none"])
```

(4c) Change the intrinsic-detection line (line 27) from:

```python
    intrinsic = a.intrinsic and a.method in ("rnd", "lpm")
```

to:

```python
    intrinsic = a.intrinsic and is_intrinsic(a.method)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `cd /data/yingte/projects/ChallengingRL-LPM/minigrid_exp && PYTHONPATH=. ../LPM_exploration/.venv/bin/python -m pytest tests/test_method_arms.py -v`
Expected: PASS (3 passed). The third test trains a short RecurrentPPO rollout (a few seconds).

- [ ] **Step 6: Commit**

```bash
cd /data/yingte/projects/ChallengingRL-LPM
git add minigrid_exp/wrappers/env_factory.py minigrid_exp/train_one.py minigrid_exp/tests/test_method_arms.py
git commit -m "feat(minigrid): rnd_lstm/lpm_lstm dispatch via base_intrinsic

env_factory picks the intrinsic wrapper from the base method; train_one
accepts the two new methods and detects intrinsic via is_intrinsic.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 5: analyze.py run-name regex

**Files:**
- Modify: `minigrid_exp/analyze.py:26-29` (`RUN_RE`)
- Test: `minigrid_exp/tests/test_analyze.py` (add cases)

**Interfaces:**
- Produces: `analyze.parse_run_name(...)` returns `method == "rnd_lstm"` / `"lpm_lstm"` for recurrent run names.

- [ ] **Step 1: Add failing tests**

Append to `minigrid_exp/tests/test_analyze.py`:

```python
def test_parse_run_name_recurrent_method():
    p = analyze.parse_run_name(
        "MiniGrid-DoorKey-5x5-v0__intrinsic_noise__rnd_lstm__seed_2__np0.1")
    assert p["method"] == "rnd_lstm" and p["np"] == "0.1" and p["seed"] == 2


def test_parse_run_name_recurrent_clean():
    p = analyze.parse_run_name(
        "MiniGrid-FourRooms-v0__intrinsic_no_noise__lpm_lstm__seed_1")
    assert p["method"] == "lpm_lstm" and p["np"] is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /data/yingte/projects/ChallengingRL-LPM/minigrid_exp && PYTHONPATH=. ../LPM_exploration/.venv/bin/python -m pytest tests/test_analyze.py -v`
Expected: FAIL — the regex matches `rnd` and leaves `_lstm__seed_2...` unmatched, so `parse_run_name` returns `None` and the test raises `TypeError`/assertion failure.

- [ ] **Step 3: Extend the method alternation (longest-first)**

In `minigrid_exp/analyze.py`, change the `RUN_RE` definition (lines 26–29) so the method group lists the `_lstm` variants before the bare ones:

```python
RUN_RE = re.compile(
    r"^(?P<env>.+?)__(?P<variant>baseline_no_noise|baseline_noise|"
    r"intrinsic_no_noise|intrinsic_noise)__"
    r"(?P<method>rnd_lstm|lpm_lstm|rnd|lpm|count|entropy|none)__seed_(?P<seed>\d+)"
    r"(?:__beta(?P<beta>[0-9.eE+-]+))?(?:__np(?P<np>[0-9.eE+-]+))?$")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /data/yingte/projects/ChallengingRL-LPM/minigrid_exp && PYTHONPATH=. ../LPM_exploration/.venv/bin/python -m pytest tests/test_analyze.py -v`
Expected: PASS (all cases, including the two new ones).

- [ ] **Step 5: Commit**

```bash
cd /data/yingte/projects/ChallengingRL-LPM
git add minigrid_exp/analyze.py minigrid_exp/tests/test_analyze.py
git commit -m "feat(minigrid): analyze parses rnd_lstm/lpm_lstm run names

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 6: Recurrent smoke test (DoorKey clean, 50k) + eval-state verification

This task produces no committed code — it is a gate that proves the recurrent pipeline composes end-to-end before the full grid. Its deliverable is a verified smoke run + a confirmation note.

**Files:**
- Reads/writes: `expr_data/minigrid/results/{models,logs}/ppo/` (smoke artifacts, removed at the end)

- [ ] **Step 1: Verify `evaluate_policy` threads the LSTM hidden state**

Run:
```bash
cd /data/yingte/projects/ChallengingRL-LPM
LPM_exploration/.venv/bin/python -c "import inspect, stable_baselines3.common.evaluation as e; src = inspect.getsource(e.evaluate_policy); print('episode_start' in src and 'states' in src)"
```
Expected: `True` (the eval loop maintains `states`/`episode_start`, so `EvalCallback` evaluates recurrent policies correctly). If `False`, add an eval helper that maintains the LSTM state across an eval episode and pass it to `EvalCallback`; otherwise continue.

- [ ] **Step 2: Run a short recurrent training cell**

Run:
```bash
cd /data/yingte/projects/ChallengingRL-LPM/minigrid_exp
PYTHONPATH=. ../LPM_exploration/.venv/bin/python train_one.py \
  --env MiniGrid-DoorKey-5x5-v0 --intrinsic --method rnd_lstm \
  --beta 0.005 --seed 1 --steps 50000 --chunk-steps 50000
```
Expected: training runs to completion with no exception; SB3 logs show `RecurrentPPO` rollout/train iterations.

- [ ] **Step 3: Confirm the smoke run produced a checkpoint and eval data**

Run:
```bash
cd /data/yingte/projects/ChallengingRL-LPM
ls -la expr_data/minigrid/results/models/ppo/MiniGrid-DoorKey-5x5-v0__intrinsic_no_noise__rnd_lstm__seed_1.zip
cat expr_data/minigrid/results/models/ppo/MiniGrid-DoorKey-5x5-v0__intrinsic_no_noise__rnd_lstm__seed_1.progress
find expr_data/minigrid/results/logs/ppo/eval -path "*rnd_lstm*" -name evaluations.npz
```
Expected: the `.zip` exists, the `.progress` file reads `50000`, and at least one `evaluations.npz` is found.

- [ ] **Step 4: Remove the smoke artifacts (so they don't pollute the real grid)**

Run:
```bash
cd /data/yingte/projects/ChallengingRL-LPM
find expr_data/minigrid/results -path "*rnd_lstm__seed_1*" -prune -print | sort   # preview
find expr_data/minigrid/results -path "*MiniGrid-DoorKey-5x5-v0__intrinsic_no_noise__rnd_lstm__seed_1*" -exec rm -rf {} +
```
Expected: the preview lists only the smoke run's files; after removal the real grid (Task 8) starts these cells fresh.

- [ ] **Step 5: Record the verification (no commit needed — this is a gate)**

Note in the execution log: "Recurrent smoke OK: RecurrentPPO + RND wrapper + EvalCallback train+eval on DoorKey-5x5 50k; evaluate_policy threads LSTM state = <True/handled>." Proceed to Task 7.

---

## Task 7: Memory-ablation figure + doc updates

**Files:**
- Modify: `minigrid_exp/make_report_figs.py` (import; `COLORS`/`LBL`; replace `m in ("rnd","lpm")` with `is_intrinsic`; add `fig_memory_ablation`; call it in `__main__`)
- Modify: `CLAUDE.md` (venv package list)
- Modify: `minigrid_exp/README.md` (verified versions + a note on the new arms/noise)

**Interfaces:**
- Consumes: `method_utils.is_intrinsic`; `cell(env, variant, method, beta, npv)` (existing helper in `make_report_figs.py`).
- Produces: `fig5_memory_ablation.png` under `expr_data/minigrid/figures/report/`.

Note: `make_report_figs.py` reads `table_final_success.csv` at import time, so it is verified by running it against real aggregated data in Task 9, not by a unit test.

- [ ] **Step 1: Add the helper import**

In `minigrid_exp/make_report_figs.py`, after `import config` (line 14), add:

```python
from method_utils import is_intrinsic
```

- [ ] **Step 2: Extend the color/label maps for the recurrent arms**

Replace the `COLORS` and `LBL` definitions (lines 52 and 54) with:

```python
COLORS = {"none": "#888888", "entropy": "#1f77b4", "rnd": "#2ca02c", "lpm": "#d62728",
          "rnd_lstm": "#98df8a", "lpm_lstm": "#ff9896"}
# Display labels: uppercase acronyms so "rnd" isn't misread as "md" at small sizes.
LBL = {"none": "none", "entropy": "entropy", "rnd": "RND", "lpm": "LPM",
       "rnd_lstm": "RND+LSTM", "lpm_lstm": "LPM+LSTM"}
```

- [ ] **Step 3: Use the helper for intrinsic detection**

In `minigrid_exp/make_report_figs.py`, replace the three inline intrinsic checks with `is_intrinsic(m)` (behavior is identical for the existing 4-method figures; this keeps the logic in one place):

- In `fig_ladder` (line ~72): `var = "intrinsic_no_noise" if is_intrinsic(m) else "baseline_no_noise"`
- In `fig_doorkey_noise` (lines ~122–123):
  ```python
      cvar = lambda m: "intrinsic_no_noise" if is_intrinsic(m) else "baseline_no_noise"
      nvar = lambda m: "intrinsic_noise" if is_intrinsic(m) else "baseline_noise"
  ```
- In `fig_fourrooms_noise` (line ~163): `var = "intrinsic_noise" if is_intrinsic(m) else "baseline_noise"`

- [ ] **Step 4: Add the memory-ablation figure**

In `minigrid_exp/make_report_figs.py`, add this function after `fig_fourrooms_noise` (before the `filmstrip` function):

```python
# --- Fig 5: memory ablation — MLP vs LSTM policy for RND and LPM ---
def fig_memory_ablation():
    """Per env, MLP vs LSTM policy for RND and LPM, clean and noisy@0.1."""
    envs = [("MiniGrid-DoorKey-5x5-v0", "DoorKey-5x5"),
            ("MiniGrid-FourRooms-v0", "FourRooms"),
            ("MiniGrid-MultiRoom-N6-v0", "MultiRoom-N6")]
    bars = ["rnd", "rnd_lstm", "lpm", "lpm_lstm"]
    panels = [("clean", "intrinsic_no_noise", np.nan),
              ("noisy 10%", "intrinsic_noise", 0.1)]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.3), sharey=True)
    for ax, (cond, var, npv) in zip(axes, panels):
        x = np.arange(len(envs)); w = 0.2
        for i, m in enumerate(bars):
            means = [cell(env, var, m, npv=npv)[0] for env, _ in envs]
            stds = [cell(env, var, m, npv=npv)[1] for env, _ in envs]
            ax.bar(x + (i - 1.5) * w, means, w, yerr=stds, capsize=2,
                   label=LBL[m], color=COLORS[m], alpha=0.85)
        ax.set_xticks(x); ax.set_xticklabels([e[1] for e in envs], fontsize=8)
        ax.set_title(cond); ax.set_ylim(0, 1.18)
        ax.legend(fontsize=7, ncol=2)
    axes[0].set_ylabel("final eval return")
    fig.suptitle("Memory ablation: MLP vs LSTM policy (RND, LPM)")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig5_memory_ablation.png"), dpi=140)
    plt.close(fig)
```

Then in the `if __name__ == "__main__":` block, add `fig_memory_ablation()` to the figure calls (right after `fig_fourrooms_noise()`):

```python
    fig_ladder(); fig_beta(); fig_doorkey_noise(); fig_fourrooms_noise()
    fig_memory_ablation()
```

- [ ] **Step 5: Syntax-check the module**

Run:
```bash
cd /data/yingte/projects/ChallengingRL-LPM/minigrid_exp
../LPM_exploration/.venv/bin/python -c "import ast; ast.parse(open('make_report_figs.py').read()); print('parse OK')"
```
Expected: `parse OK`. (Full render is verified in Task 9 once the results table exists.)

- [ ] **Step 6: Update the venv package list in `CLAUDE.md`**

In `CLAUDE.md`, find the venv bullet ending `..., `pandas`, `pytest`.` and append `sb3-contrib` so it reads `..., `pandas`, `pytest`, `sb3-contrib`.`

- [ ] **Step 7: Update `minigrid_exp/README.md`**

(7a) In the "Analysis scripts" section, after the sentence describing the run-name format, add:

```markdown
Methods include the recurrent-policy arms `rnd_lstm` / `lpm_lstm` (RND/LPM intrinsic
reward trained with an `sb3-contrib` `RecurrentPPO` `MlpLstmPolicy`); a `_lstm` suffix
selects the LSTM policy, the base method selects the intrinsic wrapper.
```

(7b) In the "Verified versions" list, add a line:

```markdown
- `sb3-contrib==2.9.0` (RecurrentPPO LSTM policy; matches sb3)
```

(7c) Add a one-line note about the noise model under the intro paragraph:

```markdown
**Observation noise** (`wrappers/noise_wrapper.py`): `noise_prob` is the fraction of
*cells* corrupted (per-cell Bernoulli); each corrupted cell is re-drawn per-channel
within MiniGrid's valid ranges (object<=10, color<=5, state<=2).
```

- [ ] **Step 8: Commit**

```bash
cd /data/yingte/projects/ChallengingRL-LPM
git add minigrid_exp/make_report_figs.py CLAUDE.md minigrid_exp/README.md
git commit -m "feat(minigrid): memory-ablation figure + docs for recurrent arms/noise

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 8: Clear stale noisy artifacts + run the experiment matrix

This task runs the ~112-cell grid. It is long-running and uses chunked checkpoint-resume (the box has an ~18-min process reaper). No code commit.

**Files:**
- Deletes (stale, old-noise): `expr_data/minigrid/results/**/*__baseline_noise__*` and `*__intrinsic_noise__*`
- Writes: `expr_data/minigrid/results/{models,logs}/ppo/` (new cells)

- [ ] **Step 1: Preview the stale noisy artifacts to delete**

The noise-model change invalidates every noisy run (all noise levels). Clean runs (`*_no_noise__*`) MUST be kept. The distinguishing tokens are `__baseline_noise__` / `__intrinsic_noise__` (note: `__intrinsic_no_noise__` does NOT contain `__intrinsic_noise__`, so these patterns are safe).

Run (preview only):
```bash
cd /data/yingte/projects/ChallengingRL-LPM
find expr_data/minigrid/results \( -name '*__baseline_noise__*' -o -name '*__intrinsic_noise__*' \) | sort | tee /tmp/stale_noisy.txt | head -40
echo "TOTAL: $(wc -l < /tmp/stale_noisy.txt)"
echo "Sanity — should be 0 (no clean runs in the list):"
grep -c '_no_noise__' /tmp/stale_noisy.txt
```
Expected: the list contains only `baseline_noise` / `intrinsic_noise` paths (models `.zip`/`.progress`, `models/best/...`, `logs/ppo/eval/...`, `logs/ppo/*.monitor.csv`, tensorboard dirs); the `_no_noise__` count is `0`.

- [ ] **Step 2: Delete the stale noisy artifacts**

Run (only after Step 1's `_no_noise__` sanity count is `0`):
```bash
cd /data/yingte/projects/ChallengingRL-LPM
find expr_data/minigrid/results \( -name '*__baseline_noise__*' -o -name '*__intrinsic_noise__*' \) -exec rm -rf {} +
find expr_data/minigrid/results \( -name '*__baseline_noise__*' -o -name '*__intrinsic_noise__*' \) | wc -l   # expect 0
```
Expected: final count `0`.

- [ ] **Step 3: Define a resume-until-done helper**

Run (defines a shell function for this session; the meta-loop re-invokes `run_grid.py` until its `--dry-run` shows 0 pending cells — surviving the reaper via chunked resume):
```bash
cd /data/yingte/projects/ChallengingRL-LPM/minigrid_exp
VENV=../LPM_exploration/.venv/bin/python
run_until_done () {
  while true; do
    n=$(PYTHONPATH=. $VENV run_grid.py "$@" --dry-run | grep -c '^\[run\]')
    echo ">>> pending cells: $n"
    [ "$n" -eq 0 ] && break
    PYTHONPATH=. $VENV run_grid.py "$@"
  done
}
```

- [ ] **Step 4: Run the clean recurrent arms (only the two new methods; existing clean runs untouched)**

Run:
```bash
cd /data/yingte/projects/ChallengingRL-LPM/minigrid_exp
run_until_done --envs MiniGrid-DoorKey-5x5-v0 --variants intrinsic_no_noise \
  --methods rnd_lstm lpm_lstm --seeds 1 2 3 4 5 6 7 8 --steps 1000000 --jobs 12
run_until_done --envs MiniGrid-FourRooms-v0 --variants intrinsic_no_noise \
  --methods rnd_lstm lpm_lstm --seeds 1 2 3 --steps 1000000 --jobs 12
run_until_done --envs MiniGrid-MultiRoom-N6-v0 --variants intrinsic_no_noise \
  --methods rnd_lstm lpm_lstm --seeds 1 2 3 --steps 2000000 --jobs 12
```
Expected: each call ends with `>>> pending cells: 0`. (Restricting `--variants intrinsic_no_noise` + `--methods rnd_lstm lpm_lstm` means existing clean `none/entropy/rnd/lpm` cells are never touched.)

- [ ] **Step 5: Run the noisy@0.1 arms (all 6 methods, regenerated under the new noise model)**

Run:
```bash
cd /data/yingte/projects/ChallengingRL-LPM/minigrid_exp
run_until_done --envs MiniGrid-DoorKey-5x5-v0 --variants baseline_noise intrinsic_noise \
  --methods rnd lpm rnd_lstm lpm_lstm --seeds 1 2 3 4 5 6 7 8 \
  --steps 1000000 --noise-probs 0.1 --jobs 12
run_until_done --envs MiniGrid-FourRooms-v0 --variants baseline_noise intrinsic_noise \
  --methods rnd lpm rnd_lstm lpm_lstm --seeds 1 2 3 \
  --steps 1000000 --noise-probs 0.1 --jobs 12
run_until_done --envs MiniGrid-MultiRoom-N6-v0 --variants baseline_noise intrinsic_noise \
  --methods rnd lpm rnd_lstm lpm_lstm --seeds 1 2 3 \
  --steps 2000000 --noise-probs 0.1 --jobs 12
```
Expected: each call ends with `>>> pending cells: 0`. (`baseline_noise` auto-runs `none`/`entropy`; `intrinsic_noise` runs the 4 intrinsic methods. All are fresh because Step 2 cleared the stale noisy artifacts.)

- [ ] **Step 6: Confirm the expected new cells exist**

Run:
```bash
cd /data/yingte/projects/ChallengingRL-LPM
echo "clean recurrent (expect 28: DoorKey 8x2 + FourRooms 3x2 + MultiRoom 3x2):"
ls expr_data/minigrid/results/models/ppo/*__intrinsic_no_noise__{rnd_lstm,lpm_lstm}__seed_*.zip 2>/dev/null | grep -v '__np' | wc -l
echo "noisy@0.1 (expect 84: 6 methods x (8+3+3) seeds):"
ls expr_data/minigrid/results/models/ppo/*__*_noise__*__np0.1.zip 2>/dev/null | grep -vE '__np0\.(2|3)' | wc -l
```
Expected: clean recurrent `28`, noisy `84` (matching the spec's matrix).

---

## Task 9: Aggregate, render figures, update FINDINGS

**Files:**
- Writes: `expr_data/minigrid/figures/` (curves, `table_final_success.csv`, `report/*.png`)
- Modify: `expr_data/minigrid/FINDINGS.md` (new numbers + memory-gain section)

- [ ] **Step 1: Aggregate eval curves + final-success table**

Run:
```bash
cd /data/yingte/projects/ChallengingRL-LPM/minigrid_exp
PYTHONPATH=. ../LPM_exploration/.venv/bin/python analyze.py
```
Expected: `loaded N eval rows` and `wrote figures + table_final_success.csv`. Confirm the table contains `rnd_lstm`/`lpm_lstm` rows:
```bash
cut -d, -f3 expr_data/minigrid/figures/table_final_success.csv | sort -u
```
Expected: includes `rnd_lstm` and `lpm_lstm`.

- [ ] **Step 2: Render report figures (incl. the memory-ablation figure)**

Run:
```bash
cd /data/yingte/projects/ChallengingRL-LPM/minigrid_exp
PYTHONPATH=. ../LPM_exploration/.venv/bin/python make_report_figs.py
ls -la expr_data/minigrid/figures/report/fig5_memory_ablation.png
```
Expected: `make_report_figs` completes; `fig5_memory_ablation.png` exists.

- [ ] **Step 3: Update `expr_data/minigrid/FINDINGS.md`**

Read `expr_data/minigrid/figures/table_final_success.csv` and add a dated section to `FINDINGS.md` recording, with the actual numbers from the table:
- the regenerated noise-robustness comparison at the new noise model (clean vs noisy@0.1 for `none`/`rnd`/`lpm`), and
- the memory gain: `rnd` vs `rnd_lstm` and `lpm` vs `lpm_lstm`, clean and noisy@0.1, per env (does an LSTM policy change the difficulty-gating or the noise-robustness ordering?).

Keep it factual — report mean±std and per-seed spread where relevant; do not assert a result the table does not show.

- [ ] **Step 4: Run the full test suite (regression check)**

Run:
```bash
cd /data/yingte/projects/ChallengingRL-LPM/minigrid_exp
PYTHONPATH=. ../LPM_exploration/.venv/bin/python -m pytest tests/ -v
```
Expected: all tests pass.

- [ ] **Step 5: Commit the analysis + findings**

```bash
cd /data/yingte/projects/ChallengingRL-LPM
git add expr_data/minigrid/FINDINGS.md 2>/dev/null || true   # expr_data is gitignored; FINDINGS may be tracked separately
git add docs/superpowers/plans/2026-06-23-minigrid-noise-and-recurrent-arms.md
git commit -m "docs(minigrid): findings for noise-model fix + recurrent arms

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```
(Note: `expr_data/` is gitignored, so figures/data are local-only; only tracked docs commit. Skip the `expr_data` add if git reports it ignored.)

---

## Self-Review

**Spec coverage:**
- Noise: cell-level Bernoulli mask, per-channel in-range redraw, seeded RNG, default 0.1, allocation-light → Task 1. ✔
- Method helpers (`base_intrinsic`/`is_recurrent`/`is_intrinsic`) → Task 2. ✔
- sb3-contrib install + RecurrentPPO + MlpLstmPolicy + efficiency knobs (shared LSTM, hidden 128) + reuse β → Tasks 3, 4. ✔
- env_factory dispatch by base method; train_one choices → Task 4. ✔
- analyze regex (longest-first) → Task 5. ✔
- make_report_figs labels/colors + helper + memory-ablation figure → Task 7. ✔
- Eval-callback LSTM-state verification + smoke test → Task 6. ✔
- Tests: noise wrapper, recurrent build/step → Tasks 1, 4. ✔
- Data: keep clean, regenerate noisy, new recurrent arms; experiment matrix (DoorKey 8 / others 3; clean new arms + noisy@0.1 all methods; sweep deferred) → Tasks 8, 9. ✔
- Docs (venv list, README) → Task 7; FINDINGS → Task 9. ✔

**Placeholder scan:** No TBD/TODO; every code step shows complete code; commands have expected output. ✔

**Type/name consistency:** `base_intrinsic`/`is_recurrent`/`is_intrinsic` defined in Task 2 and used identically in Tasks 3–5, 7. `get_algorithm_config(method)` defined in Task 3 and called with `method` at `train.py:150`. `PPO_LSTM_POLICY`/`PPO_LSTM_POLICY_KWARGS` defined in Task 3 config and imported in `train.py`. `last_cell_mask` defined and asserted in Task 1. `cell(...)`/`COLORS`/`LBL` reused consistently in Task 7. ✔
