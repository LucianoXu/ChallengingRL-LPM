# minigrid_exp — MiniGrid intrinsic-reward experiments

Incorporated + rebuilt from the private repo `github.com/JosefGh/minigrid_intrinsic_reward`
(Youssef), 2026-06-17. SB3 DQN(UCB)/PPO on flat MiniGrid observations, with a
global observation-noise wrapper and an RND intrinsic-reward wrapper. This repo's
additions: a paper-faithful LPM wrapper (`wrappers/lpm_wrapper.py`), explicit
`method`/`beta` parameters, a process-parallel grid runner (`run_grid.py`), and
analysis writing to `expr_data/minigrid/`. See `docs/minigrid_setup_analysis.md`
and `docs/SPEC.md`.

**Active learning method: PPO** (`config.ALGORITHM_NAME = "ppo"`). The project uses
PPO everywhere — on-policy, so no DQN replay-buffer / random-sampling memory pressure.
The DQN(UCB) code (`ucb_dqn.py`, `DQN_*` config) is retained but dormant.

## Analysis scripts

Use `analyze.py` for all post-run analysis — it understands the run-name format
`<env>__<variant>__<method>__seed_<n>[__beta<b>]` — and `make_report_figs.py` for the
report figures. `make_trace.py` renders a trained policy's trajectory as a GIF.

The originally-vendored `evaluate.py`, `plot_results.py` (stale: their filename parsers
predated the `__<method>__` segment and silently misparsed results) and `record_agent.py`,
`run_experiments.py` (superseded by `make_trace.py` and `run_grid.py`) were **removed**
during the 2026-06-23 cleanup; recover them from git history if ever needed.

## Verified versions (smoke-tested 2026-06-17)

- `stable-baselines3==2.9.0` (gymnasium-1.x compatible)
- `minigrid==3.1.0`
