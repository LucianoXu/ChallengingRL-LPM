# minigrid_exp — MiniGrid intrinsic-reward experiments

Incorporated + rebuilt from the private repo `github.com/JosefGh/minigrid_intrinsic_reward`
(Youssef), 2026-06-17. SB3 DQN(UCB)/PPO on flat MiniGrid observations, with a
global observation-noise wrapper and an RND intrinsic-reward wrapper. This repo's
additions: a paper-faithful LPM wrapper (`wrappers/lpm_wrapper.py`), explicit
`method`/`beta` parameters, a process-parallel grid runner (`run_grid.py`), and
analysis writing to `expr_data/minigrid/`. See `docs/minigrid_setup_analysis.md`
and `docs/SPEC.md`.

**Observation noise** (`wrappers/noise_wrapper.py`): `noise_prob` is the fraction of
*cells* corrupted (per-cell Bernoulli); each corrupted cell is re-drawn per-channel
within MiniGrid's valid ranges (object<=10, color<=5, state<=2).

**Active learning method: PPO** (`config.ALGORITHM_NAME = "ppo"`). The project uses
PPO everywhere — on-policy, so no DQN replay-buffer / random-sampling memory pressure.
The DQN(UCB) code (`ucb_dqn.py`, `DQN_*` config) is retained but dormant.

## Analysis scripts

Use `analyze.py` for all post-run analysis — it understands the run-name format
`<env>__<variant>__<method>__seed_<n>[__beta<b>]` — and `make_report_figs.py` for the
report figures. `make_trace.py` renders a single trained policy's trajectory as a GIF.
Methods include the recurrent-policy arms `rnd_lstm` / `lpm_lstm` (RND/LPM intrinsic
reward trained with an `sb3-contrib` `RecurrentPPO` `MlpLstmPolicy`); a `_lstm` suffix
selects the LSTM policy, the base method selects the intrinsic wrapper.

## Trajectory-GIF gallery (training-stage walkthroughs)

A curated, two-panel GIF gallery showing how an agent's maze walk evolves across
training stages, with the noise-induced observation anomalies made visible. See
`docs/superpowers/specs/2026-06-23-minigrid-trajectory-gifs-design.md`.

- `gif_config.py` — the 6 story-aligned configs (env × noise × method) and their
  per-stage step budgets + fixed render layout. Single source of truth.
- `make_stage_snapshots.py` — re-trains each config (one seed), dumping
  untrained/mid/final checkpoints to `results/models/ppo_gif_snapshots/`.
  `PYTHONPATH=. python make_stage_snapshots.py --jobs 6` (one subprocess/config).
- `gif_gallery.py` — renders each stage as a two-panel GIF (left: top-down maze
  with breadcrumb trail + shaded 7×7 FOV; right: the agent's egocentric view,
  where noise-corrupted cells show the true cell ghosted under TV-static), plus a
  3-stage contact strip. `PYTHONPATH=. python gif_gallery.py [--slug ...]`.
  `--smoke <run_name>` validates the renderer against an on-disk study model.
- `wrappers/ego_capture.py` — pass-through wrapper that records the exact
  (possibly noisy) image the policy saw each step.
- Output: `expr_data/minigrid/figures/gifs/<slug>/{untrained,mid,final,strip}.gif`
  + a `README.md` index. Tests: `tests/test_gif_gallery.py`.

The originally-vendored `evaluate.py`, `plot_results.py` (stale: their filename parsers
predated the `__<method>__` segment and silently misparsed results) and `record_agent.py`,
`run_experiments.py` (superseded by `make_trace.py` and `run_grid.py`) were **removed**
during the 2026-06-23 cleanup; recover them from git history if ever needed.

## Verified versions (smoke-tested 2026-06-17)

- `stable-baselines3==2.9.0` (gymnasium-1.x compatible)
- `sb3-contrib==2.9.0` (RecurrentPPO LSTM policy; matches sb3)
- `minigrid==3.1.0`
