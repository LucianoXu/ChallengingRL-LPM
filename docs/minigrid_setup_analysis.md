# Analysis of Youssef's MiniGrid repo (`JosefGh/minigrid_intrinsic_reward`)

Read 2026-06-17 from a clone of the private repo. This documents the env, the noise
model, the intrinsic-reward wiring, and what is missing for the SPEC's experiments.
Source of truth is Youssef's repo; this is our reading of it for the run side.

## What the repo is

Stable-Baselines3 agents on MiniGrid, comparing clean vs. intrinsic vs. noisy variants.

- **Algorithm:** DQN (`MlpPolicy`, flat obs) with selectable **UCB** or epsilon-greedy
  exploration (`ucb_dqn.py`), or PPO (`MlpPolicy`). Active = DQN+UCB, 5M timesteps.
- **Env matrix** (`config.py ENVIRONMENTS`): only `MiniGrid-FourRooms-v0` ("medium")
  is active; **easy** (`Empty-8x8`, `DoorKey-5x5`) and **hard** (`MultiRoom-N6`,
  `KeyCorridorS3R3`) are commented out — the difficulty ladder is scaffolded but off.
- **Seeds:** `[1, 2, 3]` (SPEC now says **8**).
- **Variants** (`config.py VARIANTS`): only `baseline_no_noise` is active; the other
  three (`intrinsic_no_noise`, `baseline_noise`, `intrinsic_noise`) are commented out.
- **DoorKey subgoal shaping** (`DoorKeySubgoalRewardWrapper`): key pickup +0.20, door
  open +0.40 — so DoorKey is *not* purely sparse when this is on (`DQN_DOORKEY_SUBGOAL_REWARDS`).

## The noise model — `wrappers/noise_wrapper.py`

`ObservationNoiseWrapper(env, noise_prob=0.10)` is a `gym.ObservationWrapper`:

```
noise_mask = rand(image.shape) < noise_prob              # per-element, iid
random_values = randint(0, 10, image.shape)              # uniform in [0,10)
image[noise_mask] = random_values[noise_mask]
```

- MiniGrid obs is the symbolic `image` tensor — by default the agent's partial 7x7x3
  view, where the 3 channels are `(object_idx, color_idx, state_idx)`.
- Each of the ~147 integers is independently replaced with prob 0.10 by a uniform int
  in `[0,10)`. Applied to **every** observation, **before** `FlatObsWrapper` flattens it.
- Minor nit: `randint(0,10)` is in-range for the object channel (~0–10) but out-of-range
  for color (0–5) and state (0–2); after `FlatObsWrapper` one-hot encoding this is
  probably harmless, just noted.

### KEY POINT: this is global observation noise, NOT a noisy-TV

The paper's noisy-TV is a **localized, navigable, (in `action_noise`) action-conditioned**
stochastic distractor: the agent moves to it and fixates because RND/ICM get a persistent
high bonus *there*. This wrapper instead corrupts **every** observation uniformly at
random, everywhere, every step. Consequences:

- There is **no spatial trap to fixate on** — the "TV-fixation" failure mode of the maze
  does not have a direct analogue here.
- It raises the RND prediction-error floor roughly **uniformly across all states**, and it
  also injects noise into the **policy's own input** (it degrades control, not just the
  bonus).

So RQ4 ("does LPM generalize to noise beyond the MiniWorld noisy-TV?") with this wrapper
tests **robustness to global observation corruption**, which is a *different* (and arguably
more pervasive) noise than the noisy-TV. The LPM-vs-RND discriminator still holds in
principle: under global noise RND keeps a high bonus (noise = novelty), so at fixed beta it
chases noise and extrinsic performance drops; LPM's bonus should stay low (unpredictable
noise → no model improvement), so it should degrade less. This is the **open decision**
flagged below.

## Intrinsic reward — `wrappers/rnd_wrapper.py`

- **Only RND is implemented.** `env_factory.make_env` raises if
  `INTRINSIC_REWARD_METHOD != "rnd"`. **LPM does not exist in this repo.**
- `RNDIntrinsicRewardWrapper` is a clean `gym.Wrapper`: in `step()` it computes a bonus =
  `reward_scale * normalized(||predictor(obs) - target(obs)||^2)`, adds it to the extrinsic
  reward, trains the predictor online, and writes the split into `info`:
  `info["extrinsic_reward"]`, `info["rnd_raw_intrinsic_reward"]`, `info["rnd_intrinsic_reward"]`.
  → the SPEC requirement "evaluate extrinsic and intrinsic separately" is **already supported**.
- **beta = `RND_REWARD_SCALE`** (`config.py`, default 0.05). The beta sweep is literally
  sweeping this scalar. RND also has its own reward running-mean-std normalization
  (`RND_NORMALIZE_REWARDS=True`), which matters for cross-method scale fairness.

## Metrics infrastructure (already present)

`train.py` wraps the env in SB3 `Monitor`/`VecMonitor` and uses `EvalCallback`
(`eval_freq=5000`, 10 episodes) → periodic eval logged to `evaluations.npz`
(timesteps vs. mean reward). That **is** the training-step-vs-reward sample-efficiency
curve the SPEC asks for — we mainly need to aggregate over 8 seeds and plot.

## Gaps to close for the SPEC experiments

1. **Implement an LPM intrinsic-reward wrapper** mirroring `RNDIntrinsicRewardWrapper`'s
   interface, reusing our paper-faithful log-space LPM
   (`LPM_exploration/Miniworld/experiments/models.py:LPMModel`, `reward_space="log"`).
   This is the main net-new engineering and the thing that joins the two halves of the project.
2. **Decide the noise model** (see open decision above): keep the global-obs-noise wrapper
   and reframe RQ4, and/or add a localized noisy-TV-style wrapper.
3. **Cross-method scale fairness:** RND normalizes its bonus and scales by beta; the LPM
   wrapper needs comparable scale handling, or beta must be swept per method (the maze
   already burned us on reward-scale mismatch — see `LPM_exploration/UPSTREAM.md`).
4. **Re-enable** the difficulty ladder (easy/hard envs) and the `noise`/`intrinsic` variants
   in `config.py`; bump `SEEDS` to 8.
5. **Wire artifacts to `expr_data/minigrid/`** (the repo writes to its own `results/`),
   per the SPEC observability requirement.
6. **Parallelism:** the runner uses a thread pool (GIL-bound for Python-side stepping);
   for the 128-core box, prefer process-level parallelism (launch many runs) like the maze
   `run_grid.py --jobs`.

## Env note

Our `LPM_exploration/.venv` has `gymnasium` 1.3.0 but **not** `minigrid` or
`stable-baselines3`. A run needs those installed (check the repo's `requirements.txt` for
the SB3 pin compatible with gymnasium 1.x).
