# MiniGrid experiment data

Produced by `minigrid_exp/run_grid.py` + `analyze.py`. Gitignored (except this file).

## Directory layout

- `results/models/dqn/<run_name>.zip` — final/best SB3 model per cell. The model `.zip`
  is the completion marker `run_grid.py` resumes on.
- `results/logs/dqn/<run_name>.monitor.csv` — per-episode training reward/length.
- `results/logs/dqn/eval/<run_name>/evaluations.npz` — periodic eval
  (keys: `timesteps`, `results` shape `[n_evals, n_episodes]`); the sample-efficiency
  curve source read by `analyze.aggregate_eval_curves`.
- `figures/fig_sample_efficiency_<env>.png` — eval return vs. training step,
  mean +/- std over 8 seeds, one line per variant/method(/beta).
- `figures/table_final_success.csv` — final-window (last 10%) mean return per cell,
  aggregated over seeds.

## Run name format

`<env>__<variant>__<method>__seed_<s>[__beta<b>]`

Example: `MiniGrid-FourRooms-v0__intrinsic_noise__lpm__seed_3__beta0.05`

- `env`: MiniGrid environment id (e.g. `MiniGrid-Empty-8x8-v0`, `MiniGrid-FourRooms-v0`)
- `variant`: one of `baseline_no_noise`, `intrinsic_no_noise`, `baseline_noise`, `intrinsic_noise`
- `method`: `rnd`, `lpm`, or `none`
- `seed`: integer seed
- `beta` (optional): intrinsic reward coefficient used in the beta sweep

## Info-dict keys

Each training step's `info` dict contains the intrinsic/extrinsic reward split for
separate tracking:
- `rnd_intrinsic_reward` or `lpm_intrinsic_reward` — intrinsic bonus at this step
- `extrinsic_reward` — original environment reward at this step
