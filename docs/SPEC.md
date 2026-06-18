# Specification for Exploration vs. Exploitation Project

Members: Yingte, Youssef.

Youssef's MiniGrid Repo: https://github.com/JosefGh/minigrid_intrinsic_reward


## Research Question to Answer

- What does the experiments in the LPM paper tell us about metrics for explorations?
- What is the corret way to mix extrinsic and intrinsic reward? What will happen if the coefficient is too small or too large?
- Does intrinsic-reward method out-perform baseline methods on RL environments with sparse reward, like minigrid?
- Does LPM method generalize to other methods with noise besides the MiniWorld environment with noisy-TV?


## Presentation Slides Structures

1. Introduction

    1. General introduction to Exploration vs. Exploitation
    2. Intrinsic Reward Methods
    3. Noisy-TV problem and LPM paper

2. Reproducting LPM paper (check report 2)

    1. Miniworld environment, explanation, demonstration
    2. Model, Setup and metrics
    3. TV-action share
    5. Coverage curve for different methods. Our experiments indicates larger variance and LPM is not the best.
    4. Coverage heatmap
    5. Conclusions and questions. 
        Coverage is not a good metric for exploration ability. Then what is a good metric for exploration?

3. Intrinsic Reward and LPM

    1. Introduce environments
    2. Model, Setup
    3. Sweep results of different beta (intrinsic reward coefficient)
    4. Success rate of different exploration methods on different environments (clean and noisy)
    5. Demonstration of traces
    6. Comparison of LPM and RND on clean and noisy performance gap.

4. Conclusion

## Work to do

2. According to the results by Yousself (see the repo and [Report2](<reports/report3/Progress meeting.pdf>)), pick the environment with appropriate difficulty. Also understand the design of the noisy observation there.
3. Conduct a parameter sweep for beta (intrinsic reward coefficient) on RND, tracking the extrinsic and intrinsic reward separately. Keep this lightweight — it is **not the main line**: the goal is only to find a coefficient in a roughly-right range that works, not an exhaustive best-beta study. Report the qualitative regime (too small → intrinsic ignored, too large → intrinsic drowns extrinsic).
4. Apply different exploration methods on different environments (clean and noisy). Ideally we will see that when difficulty goes higher, intrinsic reward methods will finally outperform.
5. We wan to verify whether LPM is robust under noise in noisy-MiniGrid environment. Compare LPM and RND on one MiniGrid environment with different ratio of observation noise.


## Requirements
- Keep all raw data organized in expr_data. Maintain observability with markdown explanations so that they can be used for further analysis.
- Use 3 seeds for each experiment, and aggregate into mean and variance. (Reduced from 8 to keep compute manageable; revisit upward for the headline LPM-vs-RND comparison if the 3-seed bands are too noisy to separate methods.)
- Diversify the MiniGrid metrics (coverage alone was shown to be a poor exploration metric in the maze). Report at least: final success rate, and **sample efficiency via a training-step vs. reward curve** (so a faster learner shows up as an earlier-rising curve even when final performance converges). Trace demonstrations as a qualitative complement. Track **extrinsic and intrinsic reward separately** (per-episode sums persisted to the monitor CSV) for the β analysis.
- Utilize parallel computing via `run_grid.py --jobs`. Under PPO each run is one process (`DummyVecEnv`, 8 envs, ~1 core); it is compute-bound (no DQN replay-buffer random-sampling), so parallelise across runs (`--jobs ≈ number of cells`, up to the core count). (The DQN path was memory-bandwidth-bound — that is no longer the binding constraint under PPO.)

## Locked experiment configuration (updated 2026-06-18)

Full running results + numbers: `expr_data/minigrid/FINDINGS.md`.

- **Learning method: PPO** everywhere (SB3 `MlpPolicy`, 8 vec-envs via `SubprocVecEnv`). DQN/UCB-DQN retired (dormant).
- **Observation: ImgObs** (147-dim 7×7×3 egocentric view), not FlatObs (whose 2835 dims were ~95% constant mission-string padding; same spatial representation, ~19× lighter).
- **Environments — one per tier:**
  - easy: `MiniGrid-DoorKey-5x5` (subgoal shaping DISABLED → purely sparse)
  - medium: `MiniGrid-FourRooms`
  - hard: `MiniGrid-MultiRoom-N6` (replaced `KeyCorridorS3R3`, which is unsolvable by a feedforward MLP — it needs key-carry memory; all methods scored 0 at 3M).
- **Exploration methods compared:** `none` (PPO baseline), `entropy` (PPO `ent_coef=0.01`, the non-intrinsic comparator), `rnd`, `lpm`. (`count` was implemented then dropped in favour of `entropy`.)
- **Variants (2×2):** clean/noisy × baseline/intrinsic; noise = 10% per-element observation corruption (global, not localized noisy-TV).
- **β (intrinsic reward_scale) — env-dependent:** β=0.05 drowns the sparse signal; usable ~0.001–0.005 on easy/medium; a *larger* β is needed on hard envs where intrinsic must drive exploration. Current config: RND 0.005, LPM 0.001 (LPM being re-swept on the hard env).
- **Budget:** medium 1M, hard 2M+ steps/run.
- **Execution:** the box has a ~18-min process reaper, so long runs use **chunked checkpoint-resume** training (`train_one --chunk-steps`, resume via `PPO.load`) driven by a shell meta-loop re-invoking `run_grid.py`; `run_grid` uses `ThreadPoolExecutor`. See `FINDINGS.md` for details.
- **Headline results (SPEC experiment list complete — full numbers + trace GIFs in `expr_data/minigrid/FINDINGS.md`):**
  1. **Intrinsic motivation is difficulty-gated:** unneeded on easy/medium (DoorKey-5x5, FourRooms — the PPO baseline already solves/explores them; intrinsic slightly hurts), **decisive on hard** — on `MultiRoom-N6` **RND solves it (eval 0.32) while baseline + entropy never reach the goal (0.0)**.
  2. **RND > LPM on clean hard exploration** (always-positive novelty drive vs LPM's signed/net-zero learning-progress signal; no β rescues LPM).
  3. **LPM is noise-robust, RND is noise-vulnerable (RQ4):** on noisy DoorKey-5x5, **RND collapses (0.82→0.035) while LPM stays robust (→0.95, like the baseline)** — a clean reproduction of the noisy-TV claim (same ordering on FourRooms; on hard MultiRoom global noise breaks everyone).