# MiniGrid matrix analysis plan

This note describes how to process the MiniGrid result matrix after adding ICM
alongside RND and LPM.

## Matrix axes

Treat each experiment cell as:

`environment x condition x method x seed`

Current primary axes:

- `environment`: DoorKey-5x5, FourRooms, MultiRoom-N6
- `condition`: clean, noisy observation
- `method`: none, entropy, RND, ICM, LPM
- `seed`: independent training seed

For sweeps, add:

- `beta`: intrinsic reward scale
- `noise_prob`: observation noise probability

## Per-seed statistic

For each seed, use the same final score definition already used in the project:

`final_return = mean(eval_return over the last 10% of evaluation points)`

This avoids overreacting to a lucky or unlucky last checkpoint.

## Aggregated statistics

For each matrix cell, report:

- `mean`: average final return over seeds
- `std`: seed-to-seed variability
- `sem`: standard error, useful for compact plots
- `median`: robust center
- `iqr`: robust spread
- `count`: number of completed seeds
- `solve_rate_0p5`: fraction of seeds with final return >= 0.5
- `zero_rate_0p05`: fraction of seeds with final return <= 0.05

The important point is to show both performance and reliability. A method with
high mean but many zero seeds is not the same story as a method with moderate
mean and all seeds solving.

## Plots to show

Use these plots in this order:

1. Matrix heatmaps: one heatmap for clean and one for noisy results, with
   environments as rows and methods as columns.
2. Difficulty ladder bars: final return by method across DoorKey, FourRooms,
   and MultiRoom, with per-seed dots over the bars.
3. Clean-vs-noisy paired bars: for each environment, compare each method under
   clean and noisy observations.
4. Sample-efficiency curves: mean evaluation return over training steps, with
   bands over seeds.
5. Solve-rate matrix: same layout as the heatmap, but values are
   `solve_rate_0p5`.
6. Beta sweep curves: for RND, ICM, and LPM, plot final return vs beta on the
   same environment.

## Interpretation checklist

When comparing RND, ICM, and LPM, answer these questions:

- Which method has the best final return?
- Which method has the highest solve rate?
- Which method has the lowest zero-rate?
- Does a method only improve on the easy environment, or does it help on the
  hard environment too?
- Under noise, does the method remain stable, or does the mean hide collapsed
  seeds?
- Does ICM behave more like RND, because both are prediction-error methods, or
  more like LPM, because its inverse-dynamics feature space filters irrelevant
  observation variation?

## Files produced by analysis

After running:

```bash
PYTHONPATH=. python minigrid_exp/analyze.py
```

the relevant outputs are:

- `expr_data/minigrid/figures/table_final_by_seed.csv`
- `expr_data/minigrid/figures/table_final_success.csv`
- `expr_data/minigrid/figures/table_matrix_stats.csv`
- `expr_data/minigrid/figures/fig_matrix_*.png`

## Server concurrency

MiniGrid PPO uses `PPO_N_ENVS = 8`, so each training cell already launches 8
environment workers. On the 128-core server, use roughly 16 concurrent cells:

```bash
PYTHONPATH=. python minigrid_exp/run_grid.py --jobs 16 --threads-per-job 1
```

This gives about `16 cells x 8 envs = 128 env workers`, while keeping
`OMP_NUM_THREADS=1` inside each cell.
