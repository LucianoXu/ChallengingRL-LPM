# Design: MiniGrid noise-model fix + recurrent-policy exploration arms

Date: 2026-06-23
Status: approved (brainstorming) — pending implementation plan

## Motivation

Two independent improvements to the sparse-reward MiniGrid study (`minigrid_exp/`):

1. **The observation-noise model is miscalibrated and semantically wrong.** The
   current wrapper corrupts each of the 147 elements of the 7×7×3 symbolic view
   independently at probability `noise_prob`, and replaces with `randint(0,10)`.
   Two problems:
   - The unit of corruption is the *channel-of-a-cell*, not the *cell*. At
     `noise_prob=0.10`, the fraction of *cells* touched is `1 - 0.9^3 ≈ 0.27`, so
     "10%" does not mean "10% of the observation is perturbed" — it is not an
     accurate measure of perturbation degree.
   - `randint(0,10)` is out of range for two of the three channels. MiniGrid
     symbolic encoding ranges are: object ∈ [0,10], color ∈ [0,5], state ∈ [0,2].
     The old noise injects color/state values the agent never sees in clean play.

2. **No memory in the experiment.** The agent is a feedforward MLP on a partially
   observable 7×7 egocentric view. We want to compare against agents with memory.
   Add two new exploration arms that train the RND and LPM intrinsic rewards with
   a recurrent (LSTM) policy, run in parallel with the existing `none`, `entropy`,
   `rnd`, `lpm` arms.

This work slots into the SPEC research questions (`docs/SPEC.md`): a cleaner,
better-calibrated noise model strengthens RQ4 (LPM noise-robustness), and the
recurrent arms add a memory axis to the difficulty-gated-exploration story.

## Feature 1 — Cell-level, range-respecting noise model

File: `minigrid_exp/wrappers/noise_wrapper.py` (rewrite `ObservationNoiseWrapper`).

The wrapper runs in `make_env` *before* `ImgObsWrapper`/`FlatObsWrapper`, so it
operates on the dict observation `obs["image"]` of shape `(H, W, 3)` (H=W=7),
uint8 symbolic encoding. Channels are (0) object id, (1) color id, (2) state id.

### New semantics

- **`noise_prob` is the fraction of cells corrupted**, not the per-element
  probability. Draw a per-cell Bernoulli mask over the `(H, W)` grid:
  `cell_mask = rng.random((H, W)) < noise_prob`. Expected corrupted-cell fraction
  is exactly `noise_prob` (decision (a): Bernoulli, not exact-count
  `round(noise_prob·H·W)` — simpler, and "fraction" reads naturally as a
  probability).
- **A corrupted cell is re-drawn as a unit**, broadcasting the single per-cell
  mask across all 3 channels. Within a corrupted cell, each channel is drawn
  independently and uniformly within its own valid range (decision (b):
  per-channel-in-range, not one coherent random object — coherence is irrelevant
  for a distractor, and per-channel is the minimal faithful change):
  - object ← `randint(0, OBJECT_MAX+1)`   (OBJECT_MAX = 10)
  - color  ← `randint(0, COLOR_MAX+1)`    (COLOR_MAX = 5)
  - state  ← `randint(0, STATE_MAX+1)`    (STATE_MAX = 2)
- The per-channel max values are read once at construction from
  `minigrid.core.constants` (`OBJECT_TO_IDX`, `COLOR_TO_IDX`, `STATE_TO_IDX`) and
  cached, so the wrapper stays correct if MiniGrid's tables change and pays no
  per-step import/lookup cost.
- **Reproducibility:** the wrapper owns a seeded `numpy.random.Generator`
  (seeded from the env seed passed through `make_env`), instead of global
  `np.random`. Falls back to an unseeded `default_rng()` when no seed is given.

### Default value (decision (c))

`noise_prob` default stays **0.10**, but now means "10% of cells" (≈ 5 of 49),
which is materially milder than the old ~27%-of-cells-touched. Still swept via
`run_grid.py --noise-probs`; the existing report sweep points {0.0, 0.1, 0.2, 0.3}
remain meaningful under the new definition.

### Efficiency

The image is 147 uint8 = trivial, but the wrapper is called every env step over
millions of steps, so keep it allocation-light:
- Reuse one `np.random.Generator` instance (created at construction).
- Cache a `channel_max` array `[OBJECT_MAX, COLOR_MAX, STATE_MAX]` at construction.
- Compute the cell mask once; index the masked cells and draw only `n_masked`
  random values per channel (not full-grid draws that are then discarded).

### Data consequence

Changing noise semantics **invalidates every existing `*_noise` run** (SPEC
headline result 3; report Figs 3 and 4). Clean (no-noise) runs are unaffected.
The noise variants must be regenerated after this change.

### Test

`tests/test_noise_wrapper.py`:
- corrupted cells obey per-channel ranges (object ≤ 10, color ≤ 5, state ≤ 2)
  over many samples;
- a cell is corrupted as a unit (within a corrupted cell all 3 channels may
  change; the mask granularity is the cell, verified by checking that the set of
  changed `(row, col)` positions is consistent across channels for a fixed seed);
- empirical corrupted-cell fraction ≈ `noise_prob` (statistical, generous band);
- `noise_prob=0.0` is a no-op; same seed ⇒ identical corruption (reproducible).

## Feature 2 — Recurrent-policy arms `rnd_lstm` and `lpm_lstm`

Add two methods. Each keeps its intrinsic-reward gym wrapper **unchanged** (the
RND/LPM wrappers add a bonus to the reward and are policy-agnostic) and swaps the
agent from `stable_baselines3.PPO` + `MlpPolicy` to
`sb3_contrib.RecurrentPPO` + `MlpLstmPolicy`.

- `rnd_lstm` = RND intrinsic bonus + LSTM policy.
- `lpm_lstm` = LPM intrinsic bonus + LSTM policy.

The clean ablation is within-method: `rnd` ↔ `rnd_lstm` and `lpm` ↔ `lpm_lstm`
(same reward shaping, MLP vs memory policy). No `none_lstm` baseline (decision
(d): out of scope).

### Method-string parsing (centralized)

To avoid the `_lstm` suffix leaking into every call site, add helpers (in
`config.py` or a small `method_utils.py`):

- `base_intrinsic(method)` → strips a trailing `_lstm` (`"rnd_lstm" → "rnd"`).
  Used by `env_factory` to pick the intrinsic wrapper and the β default.
- `is_recurrent(method)` → `method.endswith("_lstm")`. Used by `train` to pick
  the algorithm class + policy.
- `is_intrinsic(method)` → `base_intrinsic(method) in {"rnd","lpm","count"}`.
  Replaces the scattered `m in ("rnd","lpm")` checks.

### Dependency

Install `sb3-contrib` matching the pinned `stable_baselines3==2.9.0` (i.e.
`sb3-contrib==2.9.0`) into `LPM_exploration/.venv`. Record it in the venv package
notes / CLAUDE.md venv list. `RecurrentPPO` provides `MlpLstmPolicy`.

### Algorithm wiring

- `algorithms.py` / `train.py`: when `is_recurrent(method)`, use `RecurrentPPO`
  with policy `MlpLstmPolicy`; otherwise the existing `PPO` + `MlpPolicy`. The
  intrinsic wrapper is selected by `base_intrinsic(method)` in `env_factory`, so
  `rnd_lstm` builds the RND wrapper and `lpm_lstm` builds the LPM wrapper.
- β default: `env_factory` looks up `RND_REWARD_SCALE` / `LPM_REWARD_SCALE` via
  `base_intrinsic(method)`. The recurrent arms reuse the MLP β (RND 0.005,
  LPM 0.001) for the first pass (decision (e)); a recurrent-specific β re-sweep is
  flagged as future work, not done now.
- Checkpoint/resume: a given run_name always uses one class, so `RecurrentPPO`
  `.load`/`.save` resume works exactly like the PPO path. (PPO and RecurrentPPO
  checkpoints are not cross-loadable, but they never share a run_name.)

### LSTM hyperparameters (efficiency-tuned — decision (f))

`RecurrentPPO` `policy_kwargs`:
- `lstm_hidden_size = 128` (obs is 147-dim; 256 is overkill),
- `shared_lstm = True`, `enable_critic_lstm = False` (one LSTM feeds actor +
  critic instead of two separate LSTMs — roughly halves recurrent compute),
- `n_lstm_layers = 1` (default).

Reuse the existing PPO hyperparameters otherwise (`n_steps=512` = BPTT length,
`batch_size=64`, `n_epochs=10`, `gamma=0.99`, `gae_lambda=0.95`, `clip_range=0.2`,
`learning_rate=2.5e-4`; `ent_coef=0.0`).

### Efficiency notes (millions of steps)

- LSTM forward/backward on CPU is the bottleneck; `SubprocVecEnv` parallelizes env
  stepping but not the policy compute. Expect each recurrent run ≈ 2–3× the
  wall-clock of an MLP run. The efficiency knobs above (shared LSTM, hidden 128)
  are the main mitigations.
- Each run uses ~8 cores (8 subproc envs). Keep `run_grid --jobs ~12` so
  `jobs × ~8 ≤ 128` cores; chunked checkpoint-resume already handles the box's
  process reaper.

### Eval-callback correctness

SB3's `EvalCallback` uses `evaluate_policy`, which (in recent SB3) threads the
recurrent `state` and `episode_start` across eval steps. Verify this holds in the
installed sb3 2.9.0 during the smoke test; if not, supply an eval helper that
maintains the LSTM hidden state across an eval episode.

## Pipeline threading (touch-points)

- `config.py`: add LSTM hyperparams and the method-helper sets/functions.
- `algorithms.py`, `train.py`, `train_one.py`: branch class+policy on
  `is_recurrent`; add `rnd_lstm`/`lpm_lstm` to the `--method` choices; intrinsic
  detection via `is_intrinsic`/`base_intrinsic`.
- `env_factory.py`: intrinsic-wrapper dispatch and β default keyed on
  `base_intrinsic(method)`.
- `run_grid.py`: `rnd_lstm`/`lpm_lstm` are passable via `--methods`. Run-name
  tagging is unchanged.
- `analyze.py`: extend `RUN_RE` method alternation to
  `rnd_lstm|lpm_lstm|rnd|lpm|count|entropy|none` — **longest alternatives first**,
  else `rnd` matches and `_lstm` dangles.
- `make_report_figs.py`: extend `METHODS`, `COLORS`, `LBL`; replace inline
  `m in ("rnd","lpm")` with the helper; add a memory-ablation figure (per env:
  `rnd` vs `rnd_lstm`, `lpm` vs `lpm_lstm`, clean and noisy).
- `tests/`: noise-wrapper unit test (above) + a build/smoke test that
  `rnd_lstm`/`lpm_lstm` construct a `RecurrentPPO` and step a few times.

## Data implications

- Clean `none`/`entropy`/`rnd`/`lpm` runs: still valid.
- All `*_noise` runs: invalidated by the noise change → regenerate.
- `rnd_lstm`/`lpm_lstm`: new runs across variants (clean and noisy).

## Rollout (de-risk before the full grid)

1. Implement + unit-test the noise wrapper.
2. Install `sb3-contrib`; implement the recurrent wiring + helpers.
3. Smoke-test the recurrent path on DoorKey-5x5 clean at ~50k steps: confirm
   `RecurrentPPO` + intrinsic wrapper + `EvalCallback` compose and that
   `evaluate_policy` threads the LSTM state; measure fps to size the grid.
4. Regenerate the noise grid (old noise stale) and run the new recurrent arms,
   via `run_grid.py` chunked checkpoint-resume.
5. Re-run `analyze.py` + `make_report_figs.py`; add the memory-ablation figure;
   update `expr_data/minigrid/FINDINGS.md` and the LaTeX write-up.

## Out of scope

- `none_lstm` / `entropy_lstm` recurrent baselines.
- Recurrent-specific β sweep.
- Recurrent intrinsic *modules* (LSTM RND/LPM predictors) — explicitly chose
  memory-in-the-policy instead.
- Memory-requiring envs (e.g. re-introducing KeyCorridor) — keep the existing
  three-env difficulty ladder.
