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

### 5. MultiRoom-N6 beta sweep (rnd + lpm x beta in {0.005, 0.01, 0.05}, 3 seeds, 2M) — IN PROGRESS

Goal: find LPM's working beta on the hard env (does it also solve MultiRoom?) and confirm RND's
robustness across beta. Results pending. [UPDATE ON COMPLETION]

## Key takeaways so far

1. **Mixing coefficient beta matters enormously and is env-dependent.** Too large (0.05) drowns the
   sparse reward; the right value is small on easy/medium envs and larger on hard envs.
2. **Intrinsic motivation's benefit is difficulty-gated:** no help (slight harm) on medium FourRooms;
   decisive win (RND) on hard MultiRoom-N6.
3. **RND solved hard exploration where the baseline + entropy could not** — the headline result.
4. **LPM-vs-RND is not yet a fair comparison** (LPM's beta needs re-tuning per env); pending the sweep.
5. **Architecture matters:** memory-requiring tasks (KeyCorridor) are out of reach for a feedforward
   MLP regardless of exploration.

## Pending

- Finish the MultiRoom-N6 beta sweep -> fair LPM-vs-RND on the hard env.
- **Noise-robustness (RQ4):** LPM vs RND on noisy MultiRoom-N6 (observation-noise ratio sweep) —
  does LPM degrade less than RND under noise? (Requires LPM working at a good beta first.)
- (DoorKey-5x5 easy tier not yet run at the tuned beta.)
