# MiniWorld hyperparameter sweeps

This note explains how to adapt the MiniWorld reproduction experiment from the
LPM paper into controlled hyperparameter sweeps.

The new sweep utilities live in:

- `LPM_exploration/Miniworld/experiments/run_hparam_sweep.py`
- `LPM_exploration/Miniworld/experiments/summarize_hparam_sweep.py`

Outputs are written under `expr_data/miniworld/sweeps/<sweep-name>/`, keeping
large artifacts out of the tracked code tree.

## Why sweep?

The paper-faithful rerun already showed that LPM is robust to action-triggered
noisy-TV behavior, but the broader coverage result did not reproduce cleanly:
uniform random exploration covers the small extrinsic-free maze very well. A
hyperparameter sweep is useful for separating three possibilities:

1. The reproduction is sensitive to λ, entropy, update frequency, or learning
   rates.
2. The LPM mechanism is robust to noise but not meant to dominate pure coverage
   in this specific maze.
3. The paper/notebook mismatch matters: log-space LPM and legacy raw-space LPM
   may behave differently.

## Recommended workflow

For experiment 1, use the top-level bootstrap script:

```bash
python expr1.py
```

By default this is a dry-run preview. To launch the sweep:

```bash
python expr1.py --run
```

For a tiny real smoke launch through the same bootstrap path:

```bash
python expr1.py --run --steps 512 --seeds 1 --lambda-values 0.3 1.0 --entropy-values 0.03 --jobs 1 --max-runs 2
```

After summarizing experiment 1, validate the best candidates with:

```bash
python expr2.py
```

By default this previews the validation plan. It reads
`expr_data/miniworld/sweeps/expr1_lpm_core_action_noise/summary/leaderboard.csv`,
selects the top 3 configs exactly, and plans 64 fresh validation seeds
(`9..72`) for each. Launch it with:

```bash
python expr2.py --run
```

The validation output goes to
`expr_data/miniworld/sweeps/expr2_lpm_top3_action_noise_64seed/` and is
summarized automatically after successful completion. To only summarize an
existing validation folder:

```bash
python expr2.py --summarize-only
```

Useful overrides:

```bash
python expr2.py --top-k 5 --seed-start 101 --seed-count 64
python expr2.py --run --steps 512 --seed-count 2 --jobs 1 --max-runs 3
python expr2.py --run --python ./LPM_exploration/.venv/bin/python
```

Start with a dry run:

```bash
python LPM_exploration/Miniworld/experiments/run_hparam_sweep.py --preset smoke --dry-run
```

Use the project environment that already runs the MiniWorld experiments. On the
Linux compute box this usually means replacing `python` with
`./LPM_exploration/.venv/bin/python`. On Windows it is usually
`.\LPM_exploration\.venv\Scripts\python.exe` if the venv exists locally.

Then run a tiny smoke sweep:

```bash
python LPM_exploration/Miniworld/experiments/run_hparam_sweep.py --preset smoke
```

For a real diagnostic sweep, use the hardcoded `lpm_core` preset:

```bash
python LPM_exploration/Miniworld/experiments/run_hparam_sweep.py --preset lpm_core
```

The `lpm_core` preset is already configured for the 128-core server with
`jobs = 96` and `threads_per_job = 1`. This uses process-level parallelism:
one independent maze run per worker, one CPU thread per worker.

Summarize after completion:

```bash
python LPM_exploration/Miniworld/experiments/summarize_hparam_sweep.py \
  --sweep-root expr_data/miniworld/sweeps/lpm_core_action_noise
```

The main files to inspect are:

- `summary/summary_by_config.csv`: mean/std/count over seeds.
- `summary/summary_by_run.csv`: every seed separately.
- `summary/leaderboard.csv`: compact ranking by final coverage, with noisy-TV
  action share included for `action_noise`.

## Useful sweep dimensions

Keep the full factorial small. Each extra list multiplies total runs by its
length.

The hardcoded presets are defined in `HARD_CODED_SWEEPS` near the top of
`run_hparam_sweep.py`.

Current `smoke` preset:

- `steps`: `512`
- `seeds`: `1`
- `methods`: `lpm`
- `variants`: `action_noise`
- `lambda_intrinsic_values`: `0.3 1.0`
- `entropy_coef_values`: `0.03`
- `lpm_reward_space_values`: `log`
- `lpm_buffer_size_values`: `100`

Current `lpm_core` preset:

- `steps`: `50000`
- `seeds`: `1 2 3 4 5 6 7 8`
- `methods`: `lpm`
- `variants`: `action_noise`
- `lambda_intrinsic_values`: `0.1 0.3 1.0 3.0`
- `entropy_coef_values`: `0.01 0.03 0.05`
- `lpm_reward_space_values`: `log`
- `lpm_buffer_size_values`: `100`
- `jobs`: `96`
- `threads_per_job`: `1`

You can still override one piece from the command line. For example, this keeps
the `lpm_core` preset but also compares log-space LPM with legacy raw-space LPM:

```bash
python LPM_exploration/Miniworld/experiments/run_hparam_sweep.py \
  --preset lpm_core \
  --lpm-reward-space-values log raw
```

For larger grids than `lpm_core`, `--jobs 120 --threads-per-job 1` is also
reasonable on the 128-core server if memory pressure stays low.

If comparing against other intrinsic methods later, keep method-specific knobs
separate and add them deliberately:

- RND: `--rnd-lr-values 0.0003 0.001 0.003`
- ICM: `--icm-lr-values 0.0003 0.001 0.003 --icm-beta-values 0.1 0.2 0.5`
- MSE: `--mse-lr-values 0.0003 0.001 0.003`

## Interpreting the results

For `action_noise`, look at both:

- `coverage_frac_mean`: how much of the maze was covered.
- `tv_share_mean`: how often the policy chose the noisy-TV action.

The most convincing LPM setting is not necessarily the one with maximum
coverage. A good LPM setting should keep `tv_share_mean` low while maintaining
reasonable coverage. If coverage improves only by also increasing noisy-TV
fixation, that is not evidence for robust exploration.

For `nonoise` and `noisy_tv`, `tv_share_mean` is not meaningful. Use coverage
and heatmaps instead.

To make heatmaps for one promising configuration, run the existing analyzer on
that configuration directory:

```bash
python LPM_exploration/Miniworld/experiments/analyze.py \
  --results expr_data/miniworld/sweeps/lpm_core_action_noise/results/<config_slug> \
  --positions expr_data/miniworld/sweeps/lpm_core_action_noise/positions/<config_slug> \
  --figures expr_data/miniworld/sweeps/lpm_core_action_noise/figures/<config_slug>
```

Replace `<config_slug>` with the folder name from `summary/leaderboard.csv`.
