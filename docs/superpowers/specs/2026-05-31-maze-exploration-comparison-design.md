# 3D-Maze Exploration Comparison + Coverage-Heatmap Evolution — Design Spec

Date: 2026-05-31
Status: Approved (brainstorming) → ready for implementation plan
Topic: Exploration vs. Exploitation (reproducing Hou, An, Du 2026 — LPM)

## 1. Purpose

Reproduce the LPM paper's Miniworld 3D-maze exploration experiment as a controlled
comparison of intrinsic-motivation methods, and add a **new spatial statistic**: the
temporal evolution of a top-down occupancy heatmap showing *where* in the maze each
method explores *as training progresses*.

The maze emits **zero extrinsic reward** every step, so "exploration ability" is measured
purely by spatial coverage. The noisy-TV failure mode is geometric here: the random-RGB
**noise wall sits at z ∈ [8, 8.1]**, separating the lower rooms (z<8) from the far room4
(z>8.1). A method fooled by stochastic pixels should *linger near the wall*; LPM (which
rewards learning *progress*, not raw prediction error) should *cross into room4*.

This mirrors the existing Ms Pac-Man comparison infrastructure
(`LPM_exploration/Atari/experiments/`) one-for-one in shape (grid runner → per-run CSV →
analysis table + figures + a `latex_notes/` design write-up), so the two experiments read
as siblings in the report.

## 2. Research questions

- **RQ1 — coverage & noise robustness.** Under the noise wall (`noisy_tv` and
  `action_noise` vs. the clean `nonoise` control), which methods keep expanding coverage
  and which stall? Compare total coverage and *beyond-wall* coverage across
  methods × variants. Hypothesis: prediction-error methods (MSE, ICM, RND) lose coverage
  — especially beyond the wall — under noise; LPM stays high.
- **RQ2 — spatial dynamics (the new statistic).** How does the spatial *distribution* of
  exploration evolve over training? Do prediction-error methods concentrate occupancy at
  the wall over time while LPM's frontier advances into room4? Answered by the
  heatmap-evolution figure plus a `time_at_wall` metric.

## 3. Key facts established from the existing code

- The maze training in `LPM_exploration/Miniworld/*.ipynb` is a **hand-rolled A2C**
  (NOT PPO): single environment, single-epoch update every 64 steps, GAE with γ=0.99,
  λ=0.95, value-loss coef 0.5, entropy coef ≈ 0.01–0.05, max-grad-norm 0.5. Trains
  50,000 steps/run; the notebooks average over 10 runs.
- Network: Nature-style CNN (Conv 8x4-32 → 4x2-64 → 3x1-64) over (3,120,160) RGB,
  flattened (≈13056) → 512 → separate actor/critic heads.
- Intrinsic models present in the notebooks: `LearningProgressCuriosity` (LPM, η=1.0,
  reward = clip(η·E[err] − err_actual, ≤0.5)), `CuriosityModel` (ICM-style, η=0.01,
  reward = η·prediction_error), `MSEPredictionModel` (raw next-state MSE), and the
  supporting `UncertaintyPredictionModel`. **No RND** exists → we port it.
- The env already exposes per-step `info["pos"] = [x, z]`, `info["dir"]`,
  `info["visited_state"]` (count) and maintains a 72×48 visited grid, but positions are
  **only kept in memory** for end-of-run plots — never saved to disk. Saving them is the
  hook the heatmap statistic needs.
- Our own `miniworld_play/envs.py` already faithfully ports the maze geometry, the three
  variants (`NoNoiseEnv`, `NoisyTVEnv`, `ActionNoiseEnv`), the green→random transform, the
  25% sticky-action probability, and registers gymnasium IDs. It is the single source of
  truth for geometry and is reused (not re-ported).
- Stack: gymnasium 1.2.3 / miniworld 2.1.0 / stable_baselines3 2.8.0 / torch, in
  `LPM_exploration/.venv`. CPU works; MPS available; no NVIDIA GPU (Apple Silicon).

## 4. Architecture (Approach B — extract notebooks into a clean CLI trainer)

New self-contained package: `LPM_exploration/Miniworld/experiments/`.

### 4.1 Components and interfaces

- **`maze_envs.py`** — thin adapter that imports the three env classes from
  `miniworld_play/envs.py` (single source of truth for geometry) and exposes a
  `make_env(variant, seed, obs_scale)` factory returning a configured env plus the maze
  geometry constants needed downstream. Handles the obs orientation note (env returns
  HWC uint8 (120,160,3); trainer converts to CHW float). If `obs_scale != 1.0`, wraps the
  env to downscale rendered obs (e.g. 0.5 → 80×60) — a logged deviation.
- **`models.py`** — intrinsic-reward models behind a uniform interface:
  - `class IntrinsicModel` (ABC): `get_intrinsic_reward(state, next_state, action) -> float`
    and `update(states, next_states, actions) -> dict[str, float]` (returns named losses).
  - `LPMModel` (LearningProgressCuriosity, η=1.0) — wraps a prediction model + an
    uncertainty model; reward = clip(η·expected_err − actual_err, ≤0.5).
  - `ICMModel` (CuriosityModel, η=0.01) — forward-prediction error on decoded next-state.
  - `MSEModel` — raw next-state MSE prediction error.
  - `RNDModel` (**new port**) — fixed randomly-initialised target CNN (frozen) + trainable
    predictor CNN; reward = η · ||predictor(s') − target(s')||² normalised by a running std
    of intrinsic rewards. Document as a faithful new addition (not in upstream notebooks).
  - `NoneModel` — returns 0 intrinsic reward (random-walk baseline, plain A2C with no
    reward signal).
  - A `build_model(method, obs_shape, num_actions, device)` factory.
- **`a2c.py`** — `A2CAgent` extracted from the notebooks and generalised:
  - Nature-CNN actor-critic (`A2CNetwork`).
  - Rollout buffer, `compute_gae`, single-epoch `update`.
  - Accepts any `IntrinsicModel` (including `NoneModel`).
  - Combines reward as `combined = lambda_intrinsic · intrinsic` (extrinsic is 0), with the
    notebook's clipping behaviour preserved and documented.
  - Exposes per-step hooks so the trainer can record (x, z) and coverage.
- **`coverage.py`** — maze-geometry-derived helpers (pure, unit-testable):
  - `reachable_mask(nx=72, nz=48)` → boolean grid of cells whose centre lies inside any
    room rectangle (room1 x[0,4]z[0,8], room2 x[14,18]z[0,8], room3 x[0,18]z[8,8.1],
    room4 x[0,18]z[8.1,12]). The gap x∈[4,14] z∈[0,8] is unreachable and excluded.
  - `beyond_wall_mask(...)` → reachable cells in room4 (z ≥ 8.1).
  - `to_cell(x, z)` → (ix, iz) binning (ix=int(x·4), iz=int(z·4), clamped).
  - `coverage_frac`, `beyond_wall_frac`, `time_at_wall_frac` (steps with z∈[7.5,8.5]).
- **`train_maze.py`** — CLI entry point for ONE run:
  - Args: `--method {lpm,rnd,icm,mse,none} --variant {nonoise,noisy_tv,action_noise}
    --seed --steps --lambda-intrinsic --obs-scale --update-frequency --device
    --csv-log --pos-log --log-interval`.
  - Seeds torch/numpy/random/env; builds env + agent + model; trains `--steps` steps;
    maintains its own persistent occupancy grid from `info["pos"]` (independent of episode
    resets); writes per-update CSV rows and a compact `.npz` position log at the end.
- **`run_grid.py`** — iterate method × variant × seed; build `train_maze.py` commands;
  run sequentially; **resume** by skipping runs whose CSV already has enough rows. Reuses
  the Atari pattern for locating the venv python:
  `PY = <repo>/LPM_exploration/.venv/bin/python`, computed by `dirname` from the script.
  Sets `PYTORCH_ENABLE_MPS_FALLBACK=1`. Writes CSVs to `experiments/results/`, position
  logs to `experiments/positions/`, stdout to `/tmp/maze_logs/`.
- **`analyze.py`** (+ **`heatmaps.py`**) — load CSVs + position logs → table + figures into
  `experiments/figures/`.

### 4.2 Data formats

- **Per-run CSV** (`results/<run_id>.csv`), one row per logged update:
  `update, step, frames, fps, visited_count, coverage_frac, beyond_wall_frac,
  time_at_wall_frac, int_rew_mean, pred_loss, unc_loss, policy_loss, value_loss, entropy`.
  (`frames == step`; single env. Loss columns NaN where a method has no such loss.)
- **Per-run position log** (`positions/<run_id>.npz`): arrays
  `step` (int32), `x` (float32), `z` (float32), `action` (int8), `sticky` (bool).
  ~50k rows/run → tiny.
- **Run id**: `<method>-<variant>-s<seed>` (e.g. `lpm-noisy_tv-s1`).

### 4.3 Analysis outputs (`experiments/figures/`)

- `table_coverage.csv` — per (method, variant): final `coverage_frac`, `beyond_wall_frac`,
  `time_at_wall_frac` as mean ± std over seeds. "Final" = mean over the last 10% of updates
  (matches the Ms Pac-Man `final_score` convention).
- `fig_coverage_curves.png` — `coverage_frac` vs frames; one subplot per variant; one line
  per method (mean over seeds, shaded ±std), seeds aligned by interpolation.
- `fig_beyond_wall.png` — the headline noise-robustness figure: `beyond_wall_frac` per
  method, grouped by variant (clean vs noisy). Prediction: MSE/ICM/RND drop under noise,
  LPM holds.
- `fig_heatmap_evolution_<variant>.png` (one per variant) — small-multiple grid,
  **rows = method, columns = K training-progress windows** (default K=5). Two stacked
  views per figure (or two figure families): (a) **occupancy density** (steps-per-cell,
  log-scaled) and (b) **cumulative coverage frontier** (cells ever visited up to that
  window). Each panel overlays room rectangles + the noise-wall line (z=8).
- `fig_time_at_wall.png` — `time_at_wall_frac` per method × variant (the "fascination with
  noise" signal).

## 5. Compute plan — Phase 0 calibration first

Before launching the grid, run a Phase-0 calibration:
1. Confirm headless Miniworld obs generation works in a plain python process (no window),
   on this Mac, at full 160×120.
2. Measure steps/sec for env-render + A2C update, on CPU and MPS, at full res and at
   `obs_scale=0.5`.
3. Choose `--steps` and seed count to fit a few-hours total budget.

Tentative grid pending Phase-0 numbers: **5 methods × 3 variants × 2 seeds = 30 runs**,
single-env (faithful to notebooks). `--obs-scale` default 1.0 (faithful 160×120); drop to
0.5 only if throughput forces it. Per-run `--steps` likely reduced from 50k (e.g. 15k–30k)
to fit budget — the exact value is a Phase-0 output, recorded in the latex write-up.

## 6. Error handling / robustness

- `run_grid.py` resumes (skips runs whose CSV has > N rows).
- `PYTORCH_ENABLE_MPS_FALLBACK=1`; `--device auto` picks mps→cpu.
- Deterministic seeding across torch/numpy/random and `env.reset(seed=...)`.
- Trainer maintains its own persistent occupancy grid (robust to episode resets within a
  run), rather than trusting the env's grid lifecycle.

## 7. Testing

- **Phase-0 smoke**: train `none` and `lpm` for ~500 steps on `noisy_tv` — assert CSV rows
  written, position log non-empty, `coverage_frac` increases, no crash; record throughput.
- **Unit** (`coverage.py`): `to_cell` / `reachable_mask` / `beyond_wall_mask` on known
  positions → expected cells; `coverage_frac` from a synthetic position stream matches an
  independently computed count.
- **End-to-end mini-grid**: 2 methods × 1 variant × 1 seed × few-hundred steps →
  `analyze.py` produces every figure + the table without error.

## 8. Faithfulness / deviations (to log in `LPM_exploration/UPSTREAM.md`)

- A2C + intrinsic models lifted from the maze notebooks into importable modules under
  `Miniworld/experiments/` (behaviour preserved; η/λ kept at notebook values: ICM η=0.01,
  LPM η=1.0, the small reward offsets and clips documented).
- **RND newly added** (absent from notebooks): fixed random target CNN + trainable
  predictor, reward = η·normalised prediction error.
- Env geometry reused from `miniworld_play/envs.py` (no second port).
- Any `obs_scale` downscaling and any reduction of `--steps` below 50k are deviations and
  are recorded.

## 9. Deliverables

1. `LPM_exploration/Miniworld/experiments/` package
   (`maze_envs.py`, `models.py`, `a2c.py`, `coverage.py`, `train_maze.py`, `run_grid.py`,
   `analyze.py`, `heatmaps.py`, plus tests).
2. Downscaled local results: per-run CSVs + position logs → coverage curves, the
   coverage/beyond-wall table, the `time_at_wall` figure, and the heatmap-evolution figures.
3. `latex_notes/2026-05-31-maze-exploration-design.tex` write-up mirroring the Ms Pac-Man
   design doc (RQs, methods, design grid, metrics including the new heatmap statistic,
   compute budget from Phase-0, limitations), compiled to PDF.
4. `.gitignore` entries for `Miniworld/experiments/results/`, `positions/`, `figures/`.
5. `UPSTREAM.md` deviation note.

## 10. Implementation phases (for the plan)

- **Phase 0** — calibration: headless-render check + throughput measurement → set budget.
- **Phase 1** — `coverage.py` (+ unit tests), `maze_envs.py` adapter, `models.py`
  (extract LPM/ICM/MSE, port RND, NoneModel), `a2c.py` (extract + generalise).
- **Phase 2** — `train_maze.py` CLI (CSV + position logging) + smoke test; `run_grid.py`.
- **Phase 3** — run the downscaled grid locally.
- **Phase 4** — `analyze.py` + `heatmaps.py` → table + figures.
- **Phase 5** — `latex_notes/` write-up + PDF; `.gitignore`; `UPSTREAM.md`; README note.
