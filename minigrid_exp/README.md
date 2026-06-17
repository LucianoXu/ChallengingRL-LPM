# minigrid_exp — MiniGrid intrinsic-reward experiments

Incorporated + rebuilt from the private repo `github.com/JosefGh/minigrid_intrinsic_reward`
(Youssef), 2026-06-17. SB3 DQN(UCB)/PPO on flat MiniGrid observations, with a
global observation-noise wrapper and an RND intrinsic-reward wrapper. This repo's
additions: a paper-faithful LPM wrapper (`wrappers/lpm_wrapper.py`), explicit
`method`/`beta` parameters, a process-parallel grid runner (`run_grid.py`), and
analysis writing to `expr_data/minigrid/`. See `docs/minigrid_setup_analysis.md`
and `docs/SPEC.md`.

## Analysis scripts

Use `analyze.py` for all post-run analysis — it understands the run-name format
`<env>__<variant>__<method>__seed_<n>[__beta<b>]`. The vendored `evaluate.py` and
`plot_results.py` are stale: their filename parsers predate the `__<method>__`
segment and will silently misparse or skip results; do not rely on them.

## Verified versions (smoke-tested 2026-06-17)

- `stable-baselines3==2.9.0` (gymnasium-1.x compatible)
- `minigrid==3.1.0`
