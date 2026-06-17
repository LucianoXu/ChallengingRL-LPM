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
    2. Setup and metrics
    3. TV-action share
    5. Coverage curve for different methods. Our experiments indicates larger variance and LPM is not the best.
    4. Coverage heatmap
    5. Conclusions and questions. 
        Coverage is not a good metric for exploration ability. Then what is a good metric for exploration?

3. Intrinsic Reward and LPM

    1. Introduce environment and setup
    2. Sweep results of different beta (intrinsic reward coefficient)
    3. Success rate of different exploration methods on different environments (clean and noisy)
    4. Demonstration of traces
    5. Comparison of LPM and RND on clean and noisy performance gap.

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

## Locked experiment configuration (2026-06-17)

- **Learning method: PPO** everywhere (SB3 `MlpPolicy`, 8 vec-envs). DQN/UCB-DQN retired (dormant).
- **Observation: ImgObs** (147-dim 7×7×3 egocentric view), not FlatObs (whose 2835 dims were ~95% constant mission-string padding; same spatial representation, ~19× lighter).
- **Environments — one (more interesting) per tier:**
  - easy: `MiniGrid-DoorKey-5x5` (subgoal shaping DISABLED → purely sparse)
  - medium: `MiniGrid-FourRooms`
  - hard: `MiniGrid-KeyCorridorS3R3`
- **Exploration methods compared:** `none` (PPO baseline), `count` (UCB-style count-based bonus = β/sqrt(N(obs)), the non-intrinsic classical comparator), `rnd`, `lpm`.
- **Variants (2×2):** clean/noisy × baseline/intrinsic; noise = 10% per-element observation corruption (global, not localized noisy-TV).
- **Budget:** 1M PPO steps/run (revisit for KeyCorridor if undertrained).
- **Grid:** 3 envs × [2 baseline(`none`) + 2 intrinsic-variants × {count,rnd,lpm}] × 3 seeds = **72 runs**.
- **β:** RND/LPM/count all use `reward_scale=0.05` for now; per-method β sweep is a later step.