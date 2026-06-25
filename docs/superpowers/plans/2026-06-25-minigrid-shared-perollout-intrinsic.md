# MiniGrid Shared, Per-Rollout Intrinsic Reward — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace MiniGrid's per-env, per-step intrinsic-reward wrappers (RND + LPM) with a single shared model updated once per PPO rollout, then rerun every affected MiniGrid experiment and refresh figures/docs.

**Architecture:** Move the intrinsic-reward computation out of the per-subprocess `gym.Wrapper` (which forces 8 independent models, each trained every env step) into one main-process `VecEnvWrapper` that owns a single RND/LPM model, computes the bonus for all `n_envs` transitions each `step_wait`, and trains the model once per `n_steps` rollout (the PPO update cycle). Applied identically to RND and LPM so the head-to-head comparison stays apples-to-apples. The old per-env wrappers stay in the repo for the dormant DQN path + their unit tests.

**Tech Stack:** Python 3.11, `stable-baselines3==2.9.0`, `sb3-contrib==2.9.0` (RecurrentPPO), `gymnasium`, `minigrid==3.1.0`, `torch`, `numpy`, `pytest`. Venv: `LPM_exploration/.venv` (run everything via `LPM_exploration/.venv/bin/python`; `minigrid_exp` scripts need `PYTHONPATH=.` from inside `minigrid_exp/`).

## Global Constraints

- **Fix scope is BOTH methods.** Both RND and LPM move to the shared/per-rollout architecture. Never fix only one — an asymmetric fix introduces a training-schedule/architecture confound into the headline noise-robustness comparison.
- **No math LaTeX in prose/comments/docs.** Inline linear notation only: `r = g - log(mse)`, `sqrt(var+1e-8)`, `epsilon`, `g_phi`, `→`, `≈`. (Project hard rule.)
- **Paper-faithful intent unchanged.** The intrinsic-reward *formulas* (RND = predictor-vs-target MSE; LPM log-space `r = g_phi - log(MSE)`, gated until `|D| = buffer_size`) are preserved exactly. Only *where the model lives* (shared) and *when it trains* (per rollout) change.
- **Reuse, don't duplicate networks.** Import `RunningMeanStd` + `RNDNetwork` from `wrappers/rnd_wrapper.py` and `_MLP` from `wrappers/lpm_wrapper.py`. Do not redefine these.
- **PPO is the only active algorithm** (`config.ALGORITHM_NAME = "ppo"`, always `n_envs=8`). DQN (`ucb_dqn.py`) is dormant; leave its per-env-wrapper path working but untouched.
- **Compute reality:** 128-core box, no GPU, ~18-min per-process reaper. All training runs via `train_one.py` chunked checkpoint→resume (`--chunk-steps 300000`) under a shell meta-loop launched with `nohup`. `run_grid.py` uses `ThreadPoolExecutor` + `--jobs`. Use `--jobs 14` (8 envs/run × ~14 ≈ 112 cores).
- **Keep all artifacts under `expr_data/minigrid/`** (gitignored). Archive, never delete, stale data.
- **Log the deviation fix** in `LPM_exploration/UPSTREAM.md` under "Local additions / deviations" and update `CLAUDE.md` state-of-implementation.

---

## File Structure

**Create:**
- `minigrid_exp/wrappers/intrinsic_models.py` — batched, shared intrinsic models (`SharedRNDModel`, `SharedLPMModel`, `build_shared_model`). One model object owns its networks, `obs_rms`, `reward_rms`, optimizer(s), the LPM error queue `D`, and a per-rollout transition store. Methods: `reward(prev_obs, actions, next_obs) -> np.ndarray` (per-step, no-grad, also stores the transition) and `update() -> dict` (per-rollout train + clear store).
- `minigrid_exp/wrappers/intrinsic_vec_wrapper.py` — `IntrinsicVecWrapper(VecEnvWrapper)`. Drives one `SharedIntrinsicModel`: computes the bonus for all envs each `step_wait`, adds it to rewards, injects `ep_intrinsic` at episode end, and calls `model.update()` every `n_steps` step_waits.
- `minigrid_exp/tests/test_intrinsic_models.py` — unit tests for the model formulas/invariants.
- `minigrid_exp/tests/test_intrinsic_vec_wrapper.py` — unit tests for the wrapper (single shared model, per-rollout update cadence, gating, ep_intrinsic logging, reward = extrinsic + scaled intrinsic).
- `minigrid_exp/rerun_fixed_intrinsic.sh` — orchestration: archive stale rnd/lpm artifacts, then the meta-loop over all affected sub-grids.

**Modify:**
- `minigrid_exp/train.py:76-109` (`make_vector_env`) — defer rnd/lpm intrinsic to the vec wrapper; wrap the VecEnv with `IntrinsicVecWrapper` for `is_intrinsic(method)` training runs; pass `n_steps`.
- `minigrid_exp/wrappers/lpm_wrapper.py:25-40` and `wrappers/rnd_wrapper.py:53-59` — docstring note: "PPO uses the shared `IntrinsicVecWrapper`; this per-env wrapper now serves only the dormant DQN path + unit tests."
- `minigrid_exp/make_report_figs.py:119-138` (`fig_beta`) — read the FourRooms β-sweep finals from `table_final_success.csv` instead of the hardcoded `rnd=[...] / lpm=[...]` lists.
- `expr_data/minigrid/FINDINGS.md` — new numbers + a "methodology correction" subsection.
- `latex_notes/2026-06-18-minigrid-intrinsic-exploration.tex` (+ rebuilt `.pdf`) — methods description + refreshed numbers + a limitations/changelog note.
- `LPM_exploration/UPSTREAM.md`, `CLAUDE.md`, and the two project memory files — record the fix.

---

## Phase A — Implementation (TDD)

### Task A1: Shared model classes (`intrinsic_models.py`)

**Files:**
- Create: `minigrid_exp/wrappers/intrinsic_models.py`
- Test: `minigrid_exp/tests/test_intrinsic_models.py`

**Interfaces:**
- Consumes: `RunningMeanStd`, `RNDNetwork` from `wrappers.rnd_wrapper`; `_MLP` from `wrappers.lpm_wrapper`.
- Produces:
  - `SharedRNDModel(obs_dim:int, num_actions:int, reward_scale:float, learning_rate:float=1e-4, hidden_dim:int=128, output_dim:int=128, normalize_observations:bool=True, normalize_rewards:bool=True, observation_clip:float=5.0, train_epochs:int=4, train_batch:int=256, device:str="cpu", seed:int|None=None)`
  - `SharedLPMModel(obs_dim, num_actions, reward_scale, learning_rate:float=1e-3, hidden_dim:int=128, buffer_size:int=100, normalize_observations=True, normalize_rewards=True, observation_clip=5.0, train_epochs:int=4, train_batch:int=256, device="cpu", seed=None)`
  - Both expose `reward(prev_obs: np.ndarray, actions: np.ndarray, next_obs: np.ndarray) -> np.ndarray` (inputs shape `(B, obs_dim)`, `(B,)`, `(B, obs_dim)`; returns `(B,)` already scaled by `reward_scale`) and `update() -> dict`.
  - `build_shared_model(method:str, obs_dim:int, num_actions:int, reward_scale:float, device:str="cpu", seed:int|None=None) -> SharedRNDModel|SharedLPMModel` where `method in {"rnd","lpm"}` (caller passes `base_intrinsic(method)`).

- [ ] **Step 1: Write failing tests**

```python
# minigrid_exp/tests/test_intrinsic_models.py
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from wrappers.intrinsic_models import SharedRNDModel, SharedLPMModel, build_shared_model

OBS, ACT, B = 12, 3, 4

def _obs(rng): return rng.standard_normal((B, OBS)).astype(np.float32)

def test_rnd_reward_shape_and_nonneg():
    m = SharedRNDModel(OBS, ACT, reward_scale=1.0, seed=0)
    rng = np.random.default_rng(0)
    r = m.reward(_obs(rng), rng.integers(0, ACT, B), _obs(rng))
    assert r.shape == (B,)
    # raw RND bonus is a non-negative squared error; normalization keeps mean sign,
    # but with reward_scale=1 and a fresh reward_rms the first batch is finite.
    assert np.all(np.isfinite(r))

def test_rnd_update_decreases_predictor_loss():
    m = SharedRNDModel(OBS, ACT, reward_scale=1.0, train_epochs=4, train_batch=8, seed=0)
    rng = np.random.default_rng(1)
    fixed_next = _obs(rng)
    for _ in range(5):  # feed the SAME batch repeatedly, training should reduce loss
        m.reward(_obs(rng), rng.integers(0, ACT, B), fixed_next)
    first = m.update()["loss"]
    for _ in range(10):
        m.reward(_obs(rng), rng.integers(0, ACT, B), fixed_next)
        last = m.update()["loss"]
    assert last < first

def test_lpm_gated_until_buffer_fills():
    bs = 6
    m = SharedLPMModel(OBS, ACT, reward_scale=1.0, buffer_size=bs, seed=0)
    rng = np.random.default_rng(2)
    seen_nonzero = False
    # B=4 per call; buffer fills after ceil(6/4)=2 calls (8 transitions)
    rewards = []
    for _ in range(4):
        rewards.append(m.reward(_obs(rng), rng.integers(0, ACT, B), _obs(rng)))
    assert np.all(rewards[0] == 0.0), "must be gated to 0 before buffer fills"
    assert any(np.any(r != 0.0) for r in rewards[1:]), "must activate after buffer fills"

def test_lpm_update_returns_losses():
    m = SharedLPMModel(OBS, ACT, reward_scale=1.0, buffer_size=4, seed=0)
    rng = np.random.default_rng(3)
    for _ in range(3):
        m.reward(_obs(rng), rng.integers(0, ACT, B), _obs(rng))
    d = m.update()
    assert "fwd_loss" in d and "err_loss" in d

def test_build_shared_model_dispatch():
    assert isinstance(build_shared_model("rnd", OBS, ACT, 0.01), SharedRNDModel)
    assert isinstance(build_shared_model("lpm", OBS, ACT, 0.01), SharedLPMModel)
```

- [ ] **Step 2: Run, verify failure**

Run: `cd minigrid_exp && PYTHONPATH=. ../LPM_exploration/.venv/bin/python -m pytest tests/test_intrinsic_models.py -q`
Expected: FAIL — `ModuleNotFoundError: wrappers.intrinsic_models`.

- [ ] **Step 3: Implement `intrinsic_models.py`**

```python
# minigrid_exp/wrappers/intrinsic_models.py
from __future__ import annotations

import numpy as np
import torch as th
from torch import nn

from wrappers.rnd_wrapper import RunningMeanStd, RNDNetwork
from wrappers.lpm_wrapper import _MLP


def _resolve_device(device):
    if device == "auto":
        return th.device("cuda" if th.cuda.is_available() else "cpu")
    return th.device(device)


class _SharedBase:
    """Common obs-normalization + reward-normalization for shared intrinsic models.

    A single instance is owned by IntrinsicVecWrapper and fed ALL n_envs
    transitions each step (so obs_rms/reward_rms are global, unlike the old
    per-env wrappers). reward() is called every step (no-grad) and stashes the
    rollout's transitions; update() trains once per rollout and clears the stash.
    """

    def __init__(self, obs_dim, reward_scale, normalize_observations,
                 normalize_rewards, observation_clip, train_epochs, train_batch,
                 device, seed):
        self.obs_dim = int(obs_dim)
        self.reward_scale = float(reward_scale)
        self.normalize_observations = normalize_observations
        self.normalize_rewards = normalize_rewards
        self.observation_clip = observation_clip
        self.train_epochs = int(train_epochs)
        self.train_batch = int(train_batch)
        self.device = _resolve_device(device)
        if seed is not None:
            th.manual_seed(seed)
        self.obs_rms = RunningMeanStd(shape=(self.obs_dim,))
        self.reward_rms = RunningMeanStd(shape=())

    def _normalize_obs(self, flat):  # flat: (B, obs_dim) float32
        if not self.normalize_observations:
            return flat.astype(np.float32)
        z = (flat - self.obs_rms.mean) / np.sqrt(self.obs_rms.var + 1e-8)
        return np.clip(z, -self.observation_clip, self.observation_clip).astype(np.float32)

    def _normalize_reward(self, raw):  # raw: (B,) float
        if not self.normalize_rewards:
            return raw.astype(np.float32)
        self.reward_rms.update(raw.astype(np.float64))
        return (raw / np.sqrt(self.reward_rms.var + 1e-8)).astype(np.float32)

    def _minibatches(self, n):
        idx = np.random.permutation(n)
        for _ in range(self.train_epochs):
            np.random.shuffle(idx)
            for s in range(0, n, self.train_batch):
                yield idx[s:s + self.train_batch]


class SharedRNDModel(_SharedBase):
    def __init__(self, obs_dim, num_actions, reward_scale, learning_rate=1e-4,
                 hidden_dim=128, output_dim=128, normalize_observations=True,
                 normalize_rewards=True, observation_clip=5.0, train_epochs=4,
                 train_batch=256, device="cpu", seed=None):
        super().__init__(obs_dim, reward_scale, normalize_observations,
                         normalize_rewards, observation_clip, train_epochs,
                         train_batch, device, seed)
        self.target = RNDNetwork(self.obs_dim, hidden_dim, output_dim).to(self.device)
        self.predictor = RNDNetwork(self.obs_dim, hidden_dim, output_dim).to(self.device)
        for p in self.target.parameters():
            p.requires_grad = False
        self.opt = th.optim.Adam(self.predictor.parameters(), lr=learning_rate)
        self._rollout_next = []  # list of (B, obs_dim) normalized-next-obs arrays

    def reward(self, prev_obs, actions, next_obs):
        flat = np.asarray(next_obs, dtype=np.float32)
        self.obs_rms.update(flat)
        norm = self._normalize_obs(flat)
        t = th.as_tensor(norm, dtype=th.float32, device=self.device)
        with th.no_grad():
            raw = ((self.predictor(t) - self.target(t)) ** 2).mean(dim=1).cpu().numpy()
        self._rollout_next.append(norm)
        bonus = self._normalize_reward(raw.astype(np.float64))
        return (self.reward_scale * bonus).astype(np.float32)

    def update(self):
        if not self._rollout_next:
            return {"loss": 0.0}
        X = np.concatenate(self._rollout_next, axis=0)
        self._rollout_next = []
        xt = th.as_tensor(X, dtype=th.float32, device=self.device)
        with th.no_grad():
            tgt = self.target(xt)
        last = 0.0
        for mb in self._minibatches(X.shape[0]):
            loss = ((self.predictor(xt[mb]) - tgt[mb]) ** 2).mean()
            self.opt.zero_grad(); loss.backward(); self.opt.step()
            last = float(loss.item())
        return {"loss": last}


class SharedLPMModel(_SharedBase):
    def __init__(self, obs_dim, num_actions, reward_scale, learning_rate=1e-3,
                 hidden_dim=128, buffer_size=100, normalize_observations=True,
                 normalize_rewards=True, observation_clip=5.0, train_epochs=4,
                 train_batch=256, device="cpu", seed=None):
        super().__init__(obs_dim, reward_scale, normalize_observations,
                         normalize_rewards, observation_clip, train_epochs,
                         train_batch, device, seed)
        self.num_actions = int(num_actions)
        self.forward_model = _MLP(self.obs_dim + self.num_actions, hidden_dim, self.obs_dim).to(self.device)
        self.error_model = _MLP(self.obs_dim + self.num_actions, hidden_dim, 1).to(self.device)
        self.fwd_opt = th.optim.Adam(self.forward_model.parameters(), lr=learning_rate)
        self.err_opt = th.optim.Adam(self.error_model.parameters(), lr=learning_rate)
        self.buffer_size = int(buffer_size)
        self.buf = []                 # (norm_prev: (obs_dim,), action:int, mse:float)
        self._rollout = []            # (norm_prev:(obs_dim,), action:int, norm_next:(obs_dim,))

    def _sa(self, norm_obs, actions):  # norm_obs:(B,obs_dim), actions:(B,)
        a = np.zeros((norm_obs.shape[0], self.num_actions), dtype=np.float32)
        a[np.arange(norm_obs.shape[0]), np.asarray(actions, dtype=int)] = 1.0
        return th.as_tensor(np.concatenate([norm_obs, a], axis=1), dtype=th.float32, device=self.device)

    def reward(self, prev_obs, actions, next_obs):
        prev = np.asarray(prev_obs, dtype=np.float32)
        nxt = np.asarray(next_obs, dtype=np.float32)
        actions = np.asarray(actions, dtype=int)
        self.obs_rms.update(nxt)
        nprev = self._normalize_obs(prev)
        nnext = self._normalize_obs(nxt)
        sa = self._sa(nprev, actions)
        target = th.as_tensor(nnext, dtype=th.float32, device=self.device)
        with th.no_grad():
            mse = ((self.forward_model(sa) - target) ** 2).mean(dim=1).cpu().numpy()
            g = th.clamp(self.error_model(sa), -10.0, 10.0).squeeze(1).cpu().numpy()
        B = prev.shape[0]
        for i in range(B):
            self.buf.append((nprev[i], int(actions[i]), float(mse[i])))
            self._rollout.append((nprev[i], int(actions[i]), nnext[i]))
        if len(self.buf) > self.buffer_size:
            self.buf = self.buf[-self.buffer_size:]
        if len(self.buf) < self.buffer_size:
            return np.zeros(B, dtype=np.float32)   # gated (Alg 1 L6)
        raw = g - np.log(mse + 1e-6)
        bonus = self._normalize_reward(raw.astype(np.float64))
        return (self.reward_scale * bonus).astype(np.float32)

    def update(self):
        fwd_last = 0.0
        if self._rollout:
            bs = np.stack([r[0] for r in self._rollout])
            ba = np.array([r[1] for r in self._rollout], dtype=int)
            bn = np.stack([r[2] for r in self._rollout])
            self._rollout = []
            sa = self._sa(bs, ba)
            tgt = th.as_tensor(bn, dtype=th.float32, device=self.device)
            for mb in self._minibatches(bs.shape[0]):
                loss = ((self.forward_model(sa[mb]) - tgt[mb]) ** 2).mean()
                self.fwd_opt.zero_grad(); loss.backward(); self.fwd_opt.step()
                fwd_last = float(loss.item())
        err_last = 0.0
        if self.buf:
            for _ in range(self.train_epochs):
                n = min(32, len(self.buf))
                idx = np.random.choice(len(self.buf), n, replace=False)
                bs = np.stack([self.buf[i][0] for i in idx])
                ba = np.array([self.buf[i][1] for i in idx], dtype=int)
                be = np.array([self.buf[i][2] for i in idx], dtype=np.float32)
                logp = th.clamp(self.error_model(self._sa(bs, ba)), -10.0, 10.0)
                logt = th.log(th.as_tensor(be, device=self.device) + 1e-6).unsqueeze(1)
                loss = ((logp - logt) ** 2).mean()
                self.err_opt.zero_grad(); loss.backward(); self.err_opt.step()
                err_last = float(loss.item())
        return {"fwd_loss": fwd_last, "err_loss": err_last}


def build_shared_model(method, obs_dim, num_actions, reward_scale, device="cpu", seed=None):
    if method == "rnd":
        return SharedRNDModel(obs_dim, num_actions, reward_scale, device=device, seed=seed)
    if method == "lpm":
        return SharedLPMModel(obs_dim, num_actions, reward_scale, device=device, seed=seed)
    raise ValueError(f"build_shared_model: unsupported method {method!r}")
```

- [ ] **Step 4: Run tests, verify pass**

Run: `cd minigrid_exp && PYTHONPATH=. ../LPM_exploration/.venv/bin/python -m pytest tests/test_intrinsic_models.py -q`
Expected: PASS (5 tests).

- [ ] **Step 5: Commit**

```bash
git add minigrid_exp/wrappers/intrinsic_models.py minigrid_exp/tests/test_intrinsic_models.py
git commit -m "feat(minigrid): shared batched RND/LPM intrinsic models (per-rollout update)"
```

---

### Task A2: `IntrinsicVecWrapper` (`intrinsic_vec_wrapper.py`)

**Files:**
- Create: `minigrid_exp/wrappers/intrinsic_vec_wrapper.py`
- Test: `minigrid_exp/tests/test_intrinsic_vec_wrapper.py`

**Interfaces:**
- Consumes: `build_shared_model` (Task A1); `stable_baselines3.common.vec_env.VecEnvWrapper`; `make_env` from `wrappers.env_factory`.
- Produces: `IntrinsicVecWrapper(venv, model, n_steps:int)` — a `VecEnvWrapper`. `.model` is the single shared model. Adds scaled intrinsic to `rewards`; sets `info["ep_intrinsic"]` (overwriting the per-env `EpisodeRewardSplitWrapper`'s 0) at each env's episode end; calls `model.update()` every `n_steps` `step_wait`s.

- [ ] **Step 1: Write failing tests**

```python
# minigrid_exp/tests/test_intrinsic_vec_wrapper.py
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from stable_baselines3.common.vec_env import DummyVecEnv
from wrappers.env_factory import make_env
from wrappers.intrinsic_models import build_shared_model
from wrappers.intrinsic_vec_wrapper import IntrinsicVecWrapper

ENV = "MiniGrid-Empty-8x8-v0"

def _venv(n=2):
    fns = [lambda s=ENV, k=i: make_env(s, intrinsic=False, noise=False, seed=k,
                                       training=True, method="none") for i in range(n)]
    return DummyVecEnv(fns)

def _wrap(method="lpm", n_steps=8, n=2):
    venv = _venv(n)
    obs_dim = venv.observation_space.shape[0]
    num_actions = venv.action_space.n
    model = build_shared_model(method, obs_dim, num_actions, reward_scale=1.0, seed=0)
    return IntrinsicVecWrapper(venv, model, n_steps=n_steps), model

def test_single_shared_model():
    w, model = _wrap("lpm")
    assert w.model is model  # ONE model for all envs

def test_update_called_once_per_rollout(monkeypatch):
    w, model = _wrap("rnd", n_steps=5, n=2)
    calls = {"n": 0}
    orig = model.update
    monkeypatch.setattr(model, "update", lambda: (calls.__setitem__("n", calls["n"] + 1), orig())[1])
    w.reset()
    for _ in range(10):  # exactly 2 rollouts of n_steps=5
        w.step_async(np.array([w.action_space.sample() for _ in range(2)]))
        w.step_wait()
    assert calls["n"] == 2

def test_reward_is_extrinsic_plus_intrinsic():
    w, _ = _wrap("rnd", n_steps=4)
    w.reset()
    w.step_async(np.array([w.action_space.sample() for _ in range(2)]))
    _, rews, _, _ = w.step_wait()
    assert rews.shape == (2,) and np.all(np.isfinite(rews))

def test_ep_intrinsic_logged_at_episode_end():
    # MiniGrid-Empty truncates at max_steps; run long enough to end an episode.
    w, _ = _wrap("rnd", n_steps=4, n=2)
    w.reset()
    saw_ep = False
    for _ in range(600):
        w.step_async(np.array([w.action_space.sample() for _ in range(2)]))
        _, _, dones, infos = w.step_wait()
        for d, info in zip(dones, infos):
            if d:
                assert "ep_intrinsic" in info and "ep_extrinsic" in info
                saw_ep = True
        if saw_ep:
            break
    assert saw_ep
```

- [ ] **Step 2: Run, verify failure**

Run: `cd minigrid_exp && PYTHONPATH=. ../LPM_exploration/.venv/bin/python -m pytest tests/test_intrinsic_vec_wrapper.py -q`
Expected: FAIL — `ModuleNotFoundError: wrappers.intrinsic_vec_wrapper`.

- [ ] **Step 3: Implement `intrinsic_vec_wrapper.py`**

```python
# minigrid_exp/wrappers/intrinsic_vec_wrapper.py
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
```

- [ ] **Step 4: Run tests, verify pass**

Run: `cd minigrid_exp && PYTHONPATH=. ../LPM_exploration/.venv/bin/python -m pytest tests/test_intrinsic_vec_wrapper.py -q`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add minigrid_exp/wrappers/intrinsic_vec_wrapper.py minigrid_exp/tests/test_intrinsic_vec_wrapper.py
git commit -m "feat(minigrid): IntrinsicVecWrapper — shared model, per-rollout update, ep_intrinsic logging"
```

---

### Task A3: Wire the vec wrapper into `make_vector_env`

**Files:**
- Modify: `minigrid_exp/train.py:76-109`
- Test: `minigrid_exp/tests/test_intrinsic_vec_wrapper.py` (add factory-wiring test)

**Interfaces:**
- Consumes: `IntrinsicVecWrapper`, `build_shared_model`, `base_intrinsic`/`is_intrinsic` (`method_utils`), `PPO_HYPERPARAMS["n_steps"]`, `LPM_REWARD_SCALE`/`RND_REWARD_SCALE` (`config`).
- Produces: `make_vector_env(...)` returns `VecMonitor(IntrinsicVecWrapper(VecEnv))` for `training and intrinsic and is_intrinsic(method)`, and the env_fns no longer build the per-env rnd/lpm wrapper.

- [ ] **Step 1: Add failing factory test** (append to `tests/test_intrinsic_vec_wrapper.py`)

```python
def test_make_vector_env_uses_shared_wrapper():
    import train
    from config import PPO_HYPERPARAMS
    env = train.make_vector_env(ENV, intrinsic=True, noise=False, seed=0,
                                training=True, n_envs=2, log_dir=None,
                                run_name="t", method="lpm", beta=0.01)
    # VecMonitor -> IntrinsicVecWrapper -> VecEnv ; exactly one shared model.
    inner = env
    found = None
    while hasattr(inner, "venv"):
        if isinstance(inner, IntrinsicVecWrapper):
            found = inner
        inner = inner.venv
    assert found is not None and found.n_steps == PPO_HYPERPARAMS["n_steps"]
    env.close()
```

- [ ] **Step 2: Run, verify failure**

Run: `cd minigrid_exp && PYTHONPATH=. ../LPM_exploration/.venv/bin/python -m pytest tests/test_intrinsic_vec_wrapper.py::test_make_vector_env_uses_shared_wrapper -q`
Expected: FAIL — `make_vector_env` builds the old per-env wrapper / no `IntrinsicVecWrapper` in the stack.

- [ ] **Step 3: Edit `make_vector_env`** in `minigrid_exp/train.py`

Replace the body (lines 76-109) with:

```python
def make_vector_env(env_id, intrinsic, noise, seed, training, n_envs, log_dir, run_name,
                    method="rnd", beta=None, noise_prob=0.10):
    from method_utils import base_intrinsic, is_intrinsic
    from wrappers.intrinsic_models import build_shared_model
    from wrappers.intrinsic_vec_wrapper import IntrinsicVecWrapper
    from config import LPM_REWARD_SCALE, RND_REWARD_SCALE, RND_DEVICE

    # rnd/lpm intrinsic is handled by ONE shared IntrinsicVecWrapper over the
    # whole VecEnv (single model, per-rollout update). So the per-env envs are
    # built WITHOUT their own intrinsic wrapper. count (dormant) still goes per-env.
    vec_handled = bool(training and intrinsic and is_intrinsic(method))
    per_env_intrinsic = bool(intrinsic and not vec_handled)

    env_fns = []
    for env_index in range(n_envs):
        env_seed = seed + env_index

        def _make_env(env_seed=env_seed):
            return make_env(
                env_id=env_id, intrinsic=per_env_intrinsic, noise=noise,
                seed=env_seed, noise_prob=noise_prob, training=training,
                method=method, beta=beta,
            )

        env_fns.append(_make_env)

    if PPO_VEC_ENV == "subproc" and n_envs > 1:
        env = SubprocVecEnv(env_fns)
    else:
        env = DummyVecEnv(env_fns)

    if vec_handled:
        base = base_intrinsic(method)
        scale = (LPM_REWARD_SCALE if base == "lpm" else RND_REWARD_SCALE) if beta is None else beta
        model = build_shared_model(
            base, obs_dim=env.observation_space.shape[0],
            num_actions=env.action_space.n, reward_scale=scale,
            device=RND_DEVICE, seed=seed)
        env = IntrinsicVecWrapper(env, model, n_steps=PPO_HYPERPARAMS["n_steps"])

    if log_dir is not None:
        return VecMonitor(env, filename=str(log_dir / run_name),
                          info_keywords=("ep_extrinsic", "ep_intrinsic"))
    return VecMonitor(env, info_keywords=("ep_extrinsic", "ep_intrinsic"))
```

Add `PPO_HYPERPARAMS` to the `from config import (...)` block at the top of `train.py` (it already imports `PPO_N_ENVS`, `PPO_VEC_ENV`, etc.; add `PPO_HYPERPARAMS` if not already there — it IS imported, confirm).

- [ ] **Step 4: Run the full wrapper test file + the recurrent-wiring test**

Run: `cd minigrid_exp && PYTHONPATH=. ../LPM_exploration/.venv/bin/python -m pytest tests/test_intrinsic_vec_wrapper.py tests/test_recurrent_wiring.py -q`
Expected: PASS.

- [ ] **Step 5: Run the whole MiniGrid test suite (no regressions)**

Run: `cd minigrid_exp && PYTHONPATH=. ../LPM_exploration/.venv/bin/python -m pytest tests/ -q`
Expected: PASS (old `test_lpm_wrapper.py` still passes — the per-env wrapper is untouched).

- [ ] **Step 6: Add docstring notes** to `wrappers/lpm_wrapper.py` and `wrappers/rnd_wrapper.py` class docstrings:
`"NOTE: PPO now uses the shared wrappers.intrinsic_vec_wrapper.IntrinsicVecWrapper. This per-env wrapper serves the dormant DQN path and the unit tests of the reward formula."`

- [ ] **Step 7: Commit**

```bash
git add minigrid_exp/train.py minigrid_exp/tests/test_intrinsic_vec_wrapper.py minigrid_exp/wrappers/lpm_wrapper.py minigrid_exp/wrappers/rnd_wrapper.py
git commit -m "feat(minigrid): route PPO rnd/lpm through shared IntrinsicVecWrapper"
```

---

## Phase B — Smoke verification (de-risk before the big rerun)

### Task B1: Short training smoke for rnd + lpm

**Files:** none (read-only verification using `train_one.py`).

- [ ] **Step 1: Run a 60k-step smoke for each method on DoorKey clean**

Run (foreground, well under the 18-min reaper):
```bash
cd minigrid_exp
PYTHONPATH=. ../LPM_exploration/.venv/bin/python train_one.py \
  --env MiniGrid-DoorKey-5x5-v0 --intrinsic --method lpm --beta 0.005 \
  --seed 1 --steps 60000 --chunk-steps 60000
PYTHONPATH=. ../LPM_exploration/.venv/bin/python train_one.py \
  --env MiniGrid-DoorKey-5x5-v0 --intrinsic --method rnd --beta 0.005 \
  --seed 1 --steps 60000 --chunk-steps 60000
```
Expected: both complete without error; a `<run>.zip` + eval npz are written under `expr_data/minigrid/results/`.

- [ ] **Step 2: Verify the shared model is learning + bonus is sane**

Run a 2000-step probe that asserts the forward/predictor loss decreases and the intrinsic bonus is finite and nonzero after the LPM buffer fills:
```bash
cd minigrid_exp && PYTHONPATH=. ../LPM_exploration/.venv/bin/python - <<'PY'
import numpy as np
from stable_baselines3.common.vec_env import DummyVecEnv
from wrappers.env_factory import make_env
from wrappers.intrinsic_models import build_shared_model
from wrappers.intrinsic_vec_wrapper import IntrinsicVecWrapper
ENV="MiniGrid-DoorKey-5x5-v0"
venv=DummyVecEnv([lambda k=i: make_env(ENV,intrinsic=False,noise=False,seed=k,training=True,method="none") for i in range(8)])
for meth in ("rnd","lpm"):
    m=build_shared_model(meth, venv.observation_space.shape[0], venv.action_space.n, reward_scale=0.005, seed=0)
    w=IntrinsicVecWrapper(venv, m, n_steps=512); w.reset()
    losses=[]; bonuses=[]
    for t in range(1024):
        w.step_async(np.array([w.action_space.sample() for _ in range(8)]))
        _,r,_,_=w.step_wait(); bonuses.append(float(np.mean(np.abs(r))))
        if (t+1)%512==0:
            d=m.update(); losses.append(d.get("fwd_loss", d.get("loss")))
    print(meth,"losses",losses,"mean|bonus| last 200",np.mean(bonuses[-200:]))
    assert np.isfinite(bonuses[-1])
PY
```
Expected: `fwd_loss`/`loss` trends down across the two updates; mean|bonus| is finite and > 0. If LPM bonus is ~0 everywhere (forward model not learning → no LP signal), STOP and raise `INTRINSIC_TRAIN_EPOCHS` (add to model defaults) before proceeding — do not launch the big rerun on a dead signal.

- [ ] **Step 3: Commit any tuning** (only if Step 2 forced a change)

```bash
git add -A && git commit -m "chore(minigrid): smoke-tune intrinsic train epochs"
```

---

## Phase C — Archive stale data + launch the full rerun

### Task C1: Archive stale rnd/lpm artifacts

**Files:** Create `minigrid_exp/rerun_fixed_intrinsic.sh`.

- [ ] **Step 1: Write the archive+rerun script**

```bash
# minigrid_exp/rerun_fixed_intrinsic.sh
#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
PY=../LPM_exploration/.venv/bin/python
R=../expr_data/minigrid/results
STAMP=$(date +%Y%m%d-%H%M%S)
ARCH="$R/archive_pre_shared_intrinsic_$STAMP"

# 1) Move aside every stale rnd/lpm artifact (keep none/entropy — unaffected).
mkdir -p "$ARCH/models" "$ARCH/models_best" "$ARCH/logs" "$ARCH/eval"
shopt -s nullglob
for f in "$R"/models/ppo/*__{rnd,lpm,rnd_lstm,lpm_lstm}__*; do mv "$f" "$ARCH/models/" 2>/dev/null || true; done
for f in "$R"/models/ppo/*.progress; do case "$f" in *__rnd*|*__lpm*) mv "$f" "$ARCH/models/";; esac; done
for d in "$R"/models/ppo/best/*__{rnd,lpm,rnd_lstm,lpm_lstm}__*; do mv "$d" "$ARCH/models_best/" 2>/dev/null || true; done
for f in "$R"/logs/ppo/*__{rnd,lpm,rnd_lstm,lpm_lstm}__*; do mv "$f" "$ARCH/logs/" 2>/dev/null || true; done
for d in "$R"/logs/ppo/eval/*__{rnd,lpm,rnd_lstm,lpm_lstm}__*; do mv "$d" "$ARCH/eval/" 2>/dev/null || true; done
echo "archived stale rnd/lpm artifacts to $ARCH"

# 2) Meta-loop the affected sub-grids (each round re-invokes run_grid; cell_complete
#    skips finished cells, so this converges). JOBS=14 (8 envs/run).
JOBS=14
run () { PYTHONPATH=. $PY run_grid.py --jobs $JOBS "$@"; }
for r in $(seq 1 30); do
  echo "=== round $r $(date) ==="
  # 5M difficulty ladder + noisy@0.10 — DoorKey 8 seeds:
  run --envs MiniGrid-DoorKey-5x5-v0 --methods rnd lpm --seeds 1 2 3 4 5 6 7 8 --steps 5000000 --chunk-steps 300000
  # FourRooms + MultiRoom 3 seeds (clean + noisy@0.10):
  run --envs MiniGrid-FourRooms-v0 MiniGrid-MultiRoom-N6-v0 --methods rnd lpm --seeds 1 2 3 --steps 5000000 --chunk-steps 300000
  # FourRooms fine noise sweep (intrinsic_noise only, np 0.01..0.10):
  run --envs MiniGrid-FourRooms-v0 --variants intrinsic_noise --methods rnd lpm \
      --noise-probs 0.01 0.02 0.03 0.04 0.05 0.06 0.07 0.08 0.09 0.10 --seeds 1 2 3 --steps 5000000 --chunk-steps 300000
  # MultiRoom LSTM memory-ablation arms (clean + noisy@0.10):
  run --envs MiniGrid-MultiRoom-N6-v0 --methods rnd_lstm lpm_lstm --seeds 1 2 3 --steps 5000000 --chunk-steps 300000
  # MultiRoom beta robustness points (intrinsic_no_noise):
  run --envs MiniGrid-MultiRoom-N6-v0 --variants intrinsic_no_noise --methods rnd lpm \
      --betas 0.005 0.01 0.05 --seeds 1 2 3 --steps 5000000 --chunk-steps 300000
  # FourRooms beta sweep for fig2 (500k):
  run --envs MiniGrid-FourRooms-v0 --variants intrinsic_no_noise --methods rnd lpm \
      --betas 0.0005 0.001 0.005 0.01 0.05 --seeds 1 2 3 --steps 500000 --chunk-steps 250000
done
echo "rerun meta-loop done $(date)"
```

- [ ] **Step 2: Sanity-check the grid expansion is correct (dry-run count)**

Before archiving, confirm the sub-grid invocations enumerate the intended cells. Run with a no-op by inspecting `run_grid.py --help` and a single `--steps 0`-style listing if supported; otherwise eyeball against the inventory in this plan. (The inventory: DoorKey ×8, FourRooms+MultiRoom ×3 clean+noisy, FourRooms noise sweep 10×3, MultiRoom LSTM ×3, MultiRoom β{0.005,0.01,0.05}×3, FourRooms β-sweep 5×3@500k.)

- [ ] **Step 3: Commit the script, then launch detached**

```bash
git add minigrid_exp/rerun_fixed_intrinsic.sh
git commit -m "chore(minigrid): rerun orchestration for shared per-rollout intrinsic"
chmod +x minigrid_exp/rerun_fixed_intrinsic.sh
cd minigrid_exp && nohup ./rerun_fixed_intrinsic.sh > ../expr_data/minigrid/results/rerun_$(date +%Y%m%d-%H%M%S).log 2>&1 &
echo "launched; tail the log to monitor"
```

- [ ] **Step 4: Monitor to completion**

Periodically check progress: count completed cells via `.progress` sidecars reaching target steps, and watch the log tail. The full set is ~150 runs @ 5M + ~30 @ 500k; expect multiple hours. Use the Monitor tool / `ScheduleWakeup` cadence rather than blocking. Confirm no run is stuck at a chunk boundary and the meta-loop is advancing each round.

---

## Phase D — Re-analyze + regenerate figures

### Task D1: Rebuild the results table + curves

**Files:** none (runs `analyze.py`).

- [ ] **Step 1: Run analysis**

Run: `cd minigrid_exp && PYTHONPATH=. ../LPM_exploration/.venv/bin/python analyze.py --figures ../latex_notes/figs_minigrid` (use the same `--figures`/output args the repo already uses; confirm by reading `analyze.py:130-143`). Expected: `table_final_success.csv` rewritten with the new rnd/lpm numbers; curve plots regenerated.

### Task D2: De-hardcode fig2 (β sweep) and regenerate report figures

**Files:** Modify `minigrid_exp/make_report_figs.py:119-138`.

- [ ] **Step 1: Replace the hardcoded β lists with a table read**

In `fig_beta()`, replace:
```python
betas = [0.0, 0.0005, 0.001, 0.005, 0.01, 0.05]
rnd = [0.25, 0.20, 0.30, 0.30, 0.25, 0.00]
lpm = [0.25, 0.20, 0.16, 0.12, 0.09, 0.01]
```
with reads from `TAB` (the `table_final_success.csv` already loaded at module top), e.g. for each method pull `cell("MiniGrid-FourRooms-v0", "intrinsic_no_noise", m, beta=b)` for `b` in `[0.0005,0.001,0.005,0.01,0.05]` and use the b=0 baseline from the `none` row (β=0 == no intrinsic). Keep the same plot styling. (Use the existing `cell()` helper at `make_report_figs.py:70`.)

- [ ] **Step 2: Regenerate all report figures + filmstrips**

Run: `cd minigrid_exp && PYTHONPATH=. ../LPM_exploration/.venv/bin/python make_report_figs.py`
Expected: `fig1..fig6`, `strip_*` regenerate into `latex_notes/figs_minigrid/` from the new data + retrained models. Inspect each PNG; confirm the figures render and the filmstrips point at runs that still exist.

- [ ] **Step 3: (If the GIF narrative changed) regenerate stage snapshots + gallery**

Run: `cd minigrid_exp && PYTHONPATH=. ../LPM_exploration/.venv/bin/python make_stage_snapshots.py --jobs 6` then `... gif_gallery.py`. Only needed if the qualitative story (which method solves under noise) flipped; otherwise the existing GIFs stand.

- [ ] **Step 4: Commit figures**

```bash
git add minigrid_exp/make_report_figs.py latex_notes/figs_minigrid/
git commit -m "feat(minigrid): regenerate figures from shared per-rollout intrinsic reruns; de-hardcode beta sweep"
```

---

## Phase E — Update docs, writeup, memories

### Task E1: FINDINGS.md + UPSTREAM.md + CLAUDE.md

**Files:** `expr_data/minigrid/FINDINGS.md`, `LPM_exploration/UPSTREAM.md`, `CLAUDE.md`.

- [ ] **Step 1:** Add a "Methodology correction (2026-06-25)" subsection to `FINDINGS.md` describing the fix (per-step→per-rollout, 8 independent→1 shared, applied to RND and LPM) and update every numeric table/claim that changed against the new `table_final_success.csv`. Re-read the table; do not guess numbers.
- [ ] **Step 2:** In `LPM_exploration/UPSTREAM.md` "Local additions / deviations", record that the MiniGrid LPM/RND intrinsic reward moved from per-env/per-step to a shared per-rollout `IntrinsicVecWrapper`, matching the paper's per-cycle update and Miniworld's single-model design.
- [ ] **Step 3:** Update `CLAUDE.md` "State of implementation" MiniGrid paragraph: note the shared per-rollout architecture and that the per-env wrappers are now DQN-only/test-only.
- [ ] **Step 4: Commit** `git add expr_data/minigrid/FINDINGS.md LPM_exploration/UPSTREAM.md CLAUDE.md && git commit -m "docs(minigrid): record shared per-rollout intrinsic fix + refreshed findings"`

### Task E2: LaTeX writeup + PDF

**Files:** `latex_notes/2026-06-18-minigrid-intrinsic-exploration.tex` (+ `.pdf`).

- [ ] **Step 1:** Update the methods section to describe the shared, per-rollout intrinsic model (one model over 8 envs, updated once per PPO rollout = the paper's per-cycle update). Add a short "implementation correction" note explaining the prior per-step/per-env version and why both RND and LPM were re-run.
- [ ] **Step 2:** Update every figure caption/number that changed; verify each `\includegraphics` matches a regenerated PNG.
- [ ] **Step 3:** Rebuild the PDF (the repo's usual `pdflatex`/`latexmk` recipe). Confirm it compiles.
- [ ] **Step 4: Commit** `git add latex_notes/2026-06-18-minigrid-intrinsic-exploration.{tex,pdf} && git commit -m "docs(minigrid): writeup refresh for shared per-rollout intrinsic"`

### Task E3: Update project memories

**Files:** `/home/yingte/.claude/projects/-data-yingte-projects-ChallengingRL-LPM/memory/`

- [ ] **Step 1:** Update/append a memory recording that MiniGrid RND+LPM use a shared per-rollout `IntrinsicVecWrapper` (not the per-env per-step wrappers), and that the per-env wrappers are DQN-only/test-only. Add the MEMORY.md index line.

---

## Self-Review

**Spec coverage:** Claim 4 (per-step → per-rollout) → Task A1 `update()` + A2 `n_steps` gating; Claim 5 (8 independent → 1 shared) → A1 single model + A2 `.model` + A3 wiring; both fixed for RND and LPM (Global Constraint) → A1 builds both, A3 routes `is_intrinsic`. Reruns → C1 enumerates the full inventory. Figures → D1/D2 (incl. de-hardcoding fig2). Docs/writeup/memories → E1/E2/E3. No gaps.

**Placeholder scan:** All code steps contain full code; commands are exact. The only deliberately-conditional steps (B1.S3, D2.S3) state their trigger condition explicitly.

**Type consistency:** `reward(prev_obs, actions, next_obs) -> np.ndarray (B,)` and `update() -> dict` are used identically in A1 (def), A2 (`self.model.reward(...)`, `self.model.update()`), and B1 probe. `build_shared_model(method, obs_dim, num_actions, reward_scale, device, seed)` signature matches A3's call. `IntrinsicVecWrapper(venv, model, n_steps)` matches A2 def, A3 construction, and tests. `.model` / `.n_steps` attributes referenced in tests exist on the class.

**One known nuance flagged for execution:** Task B1.S2 is the gate that catches the real risk — one update per rollout could under-train the forward model and flatten LPM's learning-progress signal. `train_epochs=4` (minibatch passes per rollout) is the mitigation; B1.S2 verifies the signal is alive before committing hours of compute. Do not skip it.
