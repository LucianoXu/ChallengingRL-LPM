# MiniGrid intrinsic-reward experiments — research log

Running record of the sparse-reward MiniGrid study (PPO + intrinsic motivation).
Most recent findings first within each section. All numbers are mean over 3 seeds
unless noted. Raw data: `expr_data/minigrid/results/` (see `README.md` for layout).

## Setup

- **Algorithm:** PPO (SB3 `MlpPolicy`), 8 vectorized envs (`SubprocVecEnv`).
- **Observation:** `ImgObs` = the 7x7x3 egocentric symbolic view flattened to **147 dims**
  (NOT `FlatObs` = 2835 dims, which is ~95% constant mission-string padding — same
  spatial info, ~19x lighter).
- **Exploration methods compared:**
  - `none` — plain PPO (`ent_coef=0`), the baseline.
  - `entropy` — PPO with `ent_coef=0.01`, the *non-intrinsic* (policy-stochasticity) comparator.
  - `rnd` — Random Network Distillation intrinsic reward.
  - `lpm` — paper-faithful log-space Learning Progress Monitoring (Hou et al. 2026).
  - (`count` — count-based UCB-style bonus — was implemented then dropped in favour of `entropy`
    as the non-intrinsic comparator; the wrapper remains in the repo, unused.)
- **Intrinsic reward:** `total = extrinsic + beta * normalized(intrinsic)`. `beta` is `reward_scale`.
- **Metrics:** final eval return (deterministic, pure extrinsic) = success signal; per-episode
  extrinsic/intrinsic split in the monitor CSV; sample-efficiency via eval-return-vs-step.
- **Seeds:** 3. **Envs (difficulty ladder, one per tier):** easy `DoorKey-5x5`,
  medium `FourRooms`, hard `MultiRoom-N6` (was `KeyCorridorS3R3` — see below).

## Execution notes (important for reproduction)

- **This box (`/data/yingte`, 128-core) has a ~18-minute process reaper:** any single
  CPU-heavy process running past ~18 min gets SIGKILLed (idle/orchestrator processes survive).
  No cgroup/ulimit/systemd limit is visible; cause undiagnosed. It is the binding constraint.
- **Workaround — chunked checkpoint-resume:** `train_one.py` trains ONE chunk (`--chunk-steps`,
  ~200-300k ≈ <15 min) per process, resuming from the saved model (`PPO.load`,
  `reset_num_timesteps=False`), tracking cumulative steps in a `<run>.progress` sidecar, then
  exits. A shell **meta-loop** re-invokes `run_grid.py` ("rounds") until every cell reaches its
  total. The idle meta-loop survives the reaper; each round's children finish under it.
  Example: `for r in seq 1 N; do run_grid.py --steps 2000000 --chunk-steps 200000 ...; done`.
- **`run_grid.py` uses `ThreadPoolExecutor`** (not `ProcessPoolExecutor`, which fails to start
  workers in detached/non-TTY contexts) + per-cell `subprocess.run`. Launch detached via `nohup`.
- Per-chunk eval npz lives at `results/logs/ppo/eval/<run>/c<startstep>/evaluations.npz`;
  `analyze.py` concatenates chunks per run.

## Findings

### 1. beta sweep — FourRooms, 500k steps (clean)

Final eval return vs beta:

| method | b=0 | 0.0005 | 0.001 | 0.005 | 0.01 | 0.05 |
|---|---|---|---|---|---|---|
| rnd | 0.25 | 0.20 | 0.30 | 0.30 | 0.25 | **0.00** |
| lpm | 0.25 | 0.20 | 0.16 | 0.12 | 0.09 | **0.01** |

baseline `none` = 0.30, `entropy` = 0.23. `b=0` (no intrinsic) reproduces baseline for both (sanity).

- **beta = 0.05 catastrophically drowns the sparse extrinsic signal (-> 0)** for both methods.
  Diagnostic (per-episode sums): at b=0.05 on FourRooms the cumulative intrinsic reward was
  ~100-700x the sparse extrinsic, so the policy chased novelty and never learned the goal.
- **Usable range ~0.001-0.005.** RND peaks there; LPM declines monotonically with beta on this
  (medium) env.
- **beta is environment-dependent** (see MultiRoom): small when intrinsic is a nuisance (easy/medium),
  larger when it must drive exploration (hard). A single beta does not transfer across difficulty.

### 2. FourRooms, 1M steps (tuned beta: rnd 0.005, lpm 0.001) — clean

| none | entropy | rnd | lpm |
|---|---|---|---|
| **0.34** | 0.30 | 0.30 | 0.25 |

- **Intrinsic motivation does NOT beat the baseline on FourRooms, even at 1M.** FourRooms (medium)
  is not a hard-enough exploration problem — PPO's own policy stochasticity explores it fine
  (`none`=0.34), and any bonus (rnd/lpm) or entropy slightly *hurts*. This is the expected
  "intrinsic unneeded" regime of the difficulty ladder.

### 3. KeyCorridorS3R3, 3M steps — INFEASIBLE for this agent

All four methods scored **0.000** — not just eval, but **max training episode reward = 0.0**
across all methods, 3 seeds, 3M steps (~36M env-steps): no episode *ever* reached the object.

- **Cause: the task needs memory.** Pick up a key, *remember you carry it*, then unlock a door.
  A feedforward MLP on the image-only `ImgObs` cannot represent "I have the key" (no recurrence;
  the carried key is not in the 7x7 view). Beyond this architecture regardless of exploration.
- **Action:** dropped KeyCorridorS3R3 as the hard tier; replaced with MultiRoom-N6 (memory-free).
  (A `RecurrentPPO`/LSTM agent would be needed to use KeyCorridor.)

### 4. MultiRoom-N6, 2M steps (tuned beta: rnd 0.005, lpm 0.001) — clean  ★ HEADLINE

| method | final eval | best eval | reached goal in training? |
|---|---|---|---|
| `none` (baseline) | **0.000** | 0.000 | never |
| `entropy` | **0.000** | 0.000 | never |
| **`rnd`** | **0.316 ± 0.018** | 0.589 | yes (0.78) |
| `lpm` | 0.000 | 0.000 | never |

- **The project's key positive result:** on a genuinely hard sparse-exploration task,
  **RND (intrinsic motivation) reliably solves it (0.32, all 3 seeds, tiny variance) while the PPO
  baseline AND non-intrinsic entropy exploration never reach the goal (0.0).** Clean demonstration
  that intrinsic motivation pays off where naive exploration fails. MultiRoom-N6 is memory-free
  (doors just toggle open), so it's MLP-solvable — the only barrier is exploration depth.
- **LPM = 0.0 here, but this is likely a beta artifact, not a fair LPM verdict:** LPM's beta=0.001
  was tuned on FourRooms (where intrinsic *hurts*, so smaller was "better"); on a hard task the
  bonus must be strong enough to *drive* exploration, and 0.001 is likely too weak (RND used 0.005).
  -> motivated the MultiRoom-N6 beta sweep (below).

### 5. MultiRoom-N6 beta sweep (rnd + lpm x beta in {0.005, 0.01, 0.05}, 3 seeds, 2M)

Final eval return:

| method | b=0.005 | b=0.01 | b=0.05 |
|---|---|---|---|
| rnd | **0.28** | 0.00 | 0.00 |
| lpm | 0.00 | 0.00 | 0.00 |

- **RND solves only at b=0.005 (sweet spot).** At b=0.01/0.05 RND DOES reach the goal in stochastic
  *training* (max train reward 0.69-0.78) but fails to converge to a reliable *deterministic* policy
  (eval ~0) — too-large intrinsic keeps it chasing novelty and it never settles into exploitation.
  So even on a hard task beta has a sweet spot (~0.005): too small under-explores, too large won't converge.
- **LPM = 0 at EVERY beta** — its failure is not a beta artifact. Mechanism (from per-episode
  intrinsic magnitudes): RND's bonus is **always positive** (mean = abs, e.g. +0.28 at b=0.005) =>
  consistent directional novelty drive; LPM's bonus is **signed and nets ~0** per episode
  (mean +0.08 vs abs 0.33) => no coherent exploration push, agent behaves ~like baseline. Raising
  beta scales LPM's oscillation amplitude, not its directionality (b=0.05: mean +0.67 vs abs 4.0),
  so no beta rescues it. This is LPM's designed trade-off: the learning-progress signal goes ~0 both
  for unpredictable noise (its robustness virtue) and for already-learned local dynamics (so it
  under-explores clean hard tasks vs RND's aggressive novelty-seeking). MultiRoom shows the *cost* side.

### 6. FourRooms noise-robustness — RND vs LPM vs observation-noise ratio (1M, 3 seeds)

Final eval return vs `noise_prob` (per-element observation corruption probability):

| method | np=0.0 | np=0.1 | np=0.2 | np=0.3 |
|---|---|---|---|---|
| none | 0.32 | 0.065 | 0.045 | 0.053 |
| entropy | 0.27 | 0.076 | 0.056 | 0.051 |
| rnd | **0.32** | **0.036** | 0.024 | 0.024 |
| lpm | 0.18 | **0.073** | 0.052 | 0.045 |

- **Rank flip — LPM is relatively more noise-robust than RND (partial support for the LPM-paper claim):**
  clean (np=0), RND is best (0.32), LPM modest (0.18); under noise, **RND collapses the MOST**
  (0.32 -> 0.036, becomes the *worst*) while **LPM degrades least** and ends ABOVE RND at every noise
  level. Mechanism: noise makes every observation look novel -> RND chases noise everywhere and loses
  its lead; LPM's learning-progress signal stays ~0 on unlearnable noise -> not distracted.
- **Caveat — GLOBAL observation noise, not a localized noisy-TV:** even 10% corruption devastates ALL
  methods (all -> ~0.05) because it corrupts the policy's own perception, not just the intrinsic signal.
  So the finding is "LPM degrades LESS," not "LPM stays solved" — a relative-ordering flip near the
  failure floor. (The clean noisy-TV distractor demonstration is the earlier MiniWorld maze result.)

## Key takeaways so far

1. **Mixing coefficient beta matters enormously and is env-dependent.** Too large (0.05) drowns the
   sparse reward; the right value is small on easy/medium envs and larger on hard envs.
2. **Intrinsic motivation's benefit is difficulty-gated:** no help (slight harm) on medium FourRooms;
   decisive win (RND) on hard MultiRoom-N6.
3. **RND solved hard exploration where the baseline + entropy could not** — the headline result.
4. **RND > LPM on clean hard exploration, at EVERY beta tested** — not a tuning artifact. RND's
   always-positive novelty bonus drives exploration (sweet spot beta~0.005); LPM's signed
   learning-progress bonus nets ~0 per episode and under-explores. LPM trades exploration
   aggressiveness for its noise-robustness (the upside was shown earlier in the MiniWorld maze).
5. **Architecture matters:** memory-requiring tasks (KeyCorridor) are out of reach for a feedforward
   MLP regardless of exploration.
6. **Under observation noise (RQ4, FourRooms), LPM is relatively more robust than RND** — RND collapses
   most (loses its clean lead, becomes worst), LPM degrades least (ends above RND at every noise level).
   But *global* obs-noise breaks every method's perception (~0.05 by 10% noise), so it's "degrades less,"
   not "stays solved." Directionally supports the LPM noise-robustness thesis; the clean noisy-TV
   demonstration is the MiniWorld maze.

## Pending

- (DoorKey-5x5 easy tier not yet run at the tuned beta — likely intrinsic-unneeded like FourRooms.)
- Optional: noise-ratio sweep on the *hard* env, and/or a localized noisy-TV variant in MiniGrid for a
  cleaner distractor (vs the current global obs-noise).
- Hand off to analysis/report (figures in `figures/`, tables in `figures/table_final_success.csv`).
