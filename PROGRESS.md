# Project Plan — LPM Reproduction + Extension

> Course: *Challenging Problems in Reinforcement Learning* (SS 2026, RUB).
> Paper: Hou, An, Du 2026, *Beyond Noisy-TVs* (arXiv:2509.25438, ICLR 2026).

**Goal.** Two phases.

1. **Reproduce** the paper's pure-exploration Miniworld claims: PPO + {ICM, RND, LPM} on the 3 variants we already wrapped in `miniworld_play/envs.py`. Measure `visited_count` over training. Expected: LPM > {ICM, RND} on `noisy_tv` and `action_noise`.
2. **Extend** to a sparse-reward task with structured stochasticity: same PPO + intrinsic methods on `MiniGrid-Dynamic-Obstacles-8x8-v0`. Test whether LPM's robustness survives when the noise is a moving game-relevant entity rather than wall pixels.

---

## Stack decisions

These are the load-bearing choices — settle these first; everything else follows.

| # | Decision | Recommendation | Alternatives | Why this default |
|---|---|---|---|---|
| D1 | Env framework | **gymnasium throughout** | legacy `gym` | already in use in `miniworld_play/`; modern SB3 needs it |
| D2 | PPO implementation | **`stable_baselines3.PPO`** | port upstream PPO, hand-write | minimal code we write; battle-tested; SB3 ≥ 2.0 supports gymnasium |
| D3 | Intrinsic-reward integration | **`gymnasium.Wrapper` (per-env) that adds `r_int` to step return** | SB3 callback, custom train loop | clean separation; the wrapper holds the dynamics/error/RND networks and trains them on collected (s, a, s') tuples; works unchanged in Phase 2 |
| D4 | Intrinsic-reward model code | **extract from upstream `LPM_exploration/Atari/exploration/models/{rnd,icm,improve}.py`** | hand-rewrite, use external lib | the math we want to reproduce; upstream is the canonical reference |
| D5 | Logging | **SB3 default (TB + CSV)** | wandb, pure CSV | comes for free; TB is good enough for figures; no account needed |
| D6 | Code location | **`experiments/` top-level** | inside `LPM_exploration/` | keeps the upstream snapshot untouched per `LPM_exploration/UPSTREAM.md` |
| D7 | Vector env | **`gymnasium.vector.AsyncVectorEnv` with explicit `import miniworld_play.envs` in worker** | `SyncVectorEnv` (no parallelism) | fixes the `NameNotFound` we hit benchmarking — workers don't import registry by default |
| D8 | Compute | **Local CPU**, ≥ 8 vector workers | cluster | measured 1.7 k env-steps/sec single, → 3–5 k effective with 8 workers; 500 k-step run ≈ 2.5–10 min; full Phase 1 sweep ≈ 1.5–6 h |

If you want to change any of these, do it before P1.2 — the rest of the plan assumes the recommendation column.

---

## Phase 1 — Reproduce Miniworld

**Setup:** `miniworld_play.envs.{NoNoiseEnv, ActionNoiseEnv, NoisyTVEnv}` (already gymnasium-registered).
**Hypothesis:** on `noisy_tv` and `action_noise`, `visited_count(t)` curve: LPM keeps rising; ICM and RND plateau early. On `nonoise`: all three roughly comparable.

- [ ] **P1.1 — Extract intrinsic models into `experiments/intrinsic/`**
  Files to extract from `LPM_exploration/Atari/exploration/models/`: `rnd.py`, `icm.py`, `improve.py` (LPM forward + error predictor), `common.py`. Keep upstream files untouched; copy + rename our versions to `experiments/intrinsic/{rnd,icm,lpm}.py`. Each module exposes a single class with `train_step(batch)` and `intrinsic_reward(obs, action, next_obs)` methods. Acceptance: each model instantiates on CPU and runs a 1-batch forward/backward smoke test.

- [ ] **P1.2 — Write `experiments/phase1/train.py`**
  Uses `stable_baselines3.PPO` + an `IntrinsicRewardWrapper` (`experiments/phase1/wrapper.py`) that mixes `r_total = r_ext + λ · r_int`. CLI: `--variant {nonoise,action_noise,noisy_tv} --intrinsic {none,icm,rnd,lpm} --steps N --seed S --workers W --out DIR`. Acceptance: 10 000 env-step end-to-end run on `noisy_tv × lpm` produces an SB3 `progress.csv` with non-trivial `r_int` column.

- [ ] **P1.3 — Production sweep**
  3 variants × 4 methods × 3 seeds = 36 runs × 500 k steps. Driver: `experiments/phase1/run.sh`. Estimated 1.5–6 h total. Deliverable: SB3 logs in `experiments/phase1/runs/`. Plot script produces `visited_count vs env_step` figure (4 lines per variant, mean ± SD over seeds).

- [ ] **P1.4 — Phase 1 writeup**
  `reports/phase1.md`, 2–3 pages. Setup + main figure + paired t-test on final `visited_count` + deviations from paper.

---

## Phase 2 — Extend to MiniGrid Dynamic-Obstacles

**Setup:** `MiniGrid-Dynamic-Obstacles-8x8-v0`. 4 balls perform a uniform random step each env step; collision = terminal + 0 reward; goal-reach = `1 - 0.9 · (step/max_steps)`. Plus `MiniGrid-DoorKey-8x8-v0` as the deterministic control.

**Hypotheses:**
- **H1 (control).** All four methods comparable on DoorKey-8x8 (deterministic; sparse reward is the dominant difficulty).
- **H2 (curiosity trap).** On Dynamic-Obstacles-8x8, PPO+ICM and PPO+RND die more often than vanilla PPO — the random balls attract them.
- **H3 (LPM robust).** PPO+LPM's death rate matches vanilla PPO — LPM correctly identifies ball positions as unlearnable and stops emitting bonus.

- [ ] **P2.1 — Add MiniGrid + RGB-obs adapter**
  Install: `uv pip install --python LPM_exploration/.venv/bin/python minigrid`. Write `experiments/phase2/envs.py` with a gymnasium `ObservationWrapper` that renders to 84×84 RGB (parity with Phase 1's CNN). Acceptance: 20-step random rollout on both envs returns `(84, 84, 3) uint8`.

- [ ] **P2.2 — Adapt `train.py` to MiniGrid**
  Copy `experiments/phase1/train.py` → `experiments/phase2/train.py`. Change env builder; add `--env {doorkey,dynobs}` and `--lambda-intrinsic FLOAT` flags. The intrinsic-reward wrapper from P1.2 is reused unchanged. Acceptance: 5 000-step run on `dynobs × lpm` produces non-empty SB3 logs.

- [ ] **P2.3 — Sweep**
  4 methods × 3 seeds × 2 envs = 24 runs × 1 M steps. Estimated 1–4 h total. Deliverable: three figures (ep_return convergence, death rate over training, exploration coverage scatter). Welch's t-test on per-seed final-50k death rate verdicts H1/H2/H3.

- [ ] **P2.4 — Phase 2 writeup**
  `reports/phase2.md`, 3–4 pages.

---

## Phase 3 — Final report + slides

- [ ] **P3.1 — `reports/final.md`** — 8 pages: motivation, related work, Phase 1 reproduction, Phase 2 extension, limitations, future work. Cite the OpenReview concerns this project addresses.
- [ ] **P3.2 — `reports/final_slides.{key,pdf}`** — 15-min talk; one figure per main claim; live demo of `miniworld_play/play.py` if time allows.

---

## Risks (top 3 only)

| risk | mitigation |
|---|---|
| Phase 2 H3 fails (LPM also fooled by moving balls) | Still a publishable negative result; reviewer CKoM hinted at it; write up as "LPM's robustness limited to unstructured noise" |
| Upstream intrinsic models hard-coded for CUDA / 84×84 Atari obs and don't accept 160×120 Miniworld obs | P1.1 surfaces this; if blocking, hand-rewrite the offending model from the paper's equations (~50 LoC each) |
| SB3 PPO + custom intrinsic wrapper has subtle bugs (e.g. wrong gradient scaling between ext and int) | Add a "vanilla PPO with `λ=0`" check that reproduces SB3 baseline behaviour exactly before adding intrinsic |

---

`[ ]` not started · `[~]` in progress · `[x]` done. Edit in place; commit each status flip.
