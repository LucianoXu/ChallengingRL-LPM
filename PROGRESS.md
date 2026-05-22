# Project Plan: LPM Reproduction + Extrinsic-Reward Extension

> Course: *Challenging Problems in Reinforcement Learning* (SS 2026, RUB).
> Paper: Hou, An, Du 2026 — *Beyond Noisy-TVs: Noise-Robust Exploration via Learning Progress Monitoring* (arXiv:2509.25438, ICLR 2026 Poster).

**Goal.** Reproduce the paper's pure-exploration Miniworld results, then extend LPM to a sparse-reward task with structured stochasticity (MiniGrid `Dynamic-Obstacles-8x8`).

**Architecture.** Build a single PPO training script that takes `--intrinsic {none,icm,rnd,lpm}` as a switch. Plug the paper's existing intrinsic-reward model implementations (in `LPM_exploration/Atari/exploration/models/`) into our own training loop. Use the env layer we already wrote in `miniworld_play/envs.py` for Phase 1; add a thin MiniGrid adapter for Phase 2.

**Tech stack.** Python 3.11, PyTorch, gymnasium, miniworld, minigrid, PPO (ported from upstream `LPM_exploration/Atari/exploration/algo/ppo.py`), CSV logging.

**Compute constraint.** Apple Silicon CPU/MPS only. No GPU cluster. Drives all scoping decisions (number of seeds, step counts, parallel workers).

---

## Phase 1 — Reproduce the paper's Miniworld exploration claims

**Setup.** Use our `miniworld_play.envs.{NoNoiseEnv, ActionNoiseEnv, NoisyTVEnv}` as the environment. All three variants are already gymnasium-registered and tested.

**Hypothesis to confirm.** On `noisy_tv` and `action_noise`, LPM's `visited_count` curve keeps climbing while ICM's and RND's plateau or degrade — the curiosity-trap failure mode is real and LPM avoids it.

### Tasks

- [ ] **P1.1 — Audit upstream intrinsic-reward implementations**
  - Read: `LPM_exploration/Atari/exploration/models/{rnd.py, icm.py, improve.py, common.py}` and `LPM_exploration/Atari/exploration/algo/ppo.py`.
  - Document each model's input shape, forward signature, loss target, optimiser, and any CUDA-only ops.
  - Deliverable: `notes/phase1_audit.md`. One short paragraph per file. Flag any porting blockers.

- [ ] **P1.2 — Write a unified PPO + intrinsic training script**
  - File: `experiments/phase1/train.py`. Self-contained.
  - CLI: `--variant {nonoise,action_noise,noisy_tv} --intrinsic {none,icm,rnd,lpm} --steps N --seed S --workers W --out DIR`.
  - PPO: actor-critic CNN (Nature-DQN style, matching paper §A), GAE, clipped objective.
  - Vectorised env via `gymnasium.vector.AsyncVectorEnv`, default `--workers 8` (tune for this Mac).
  - Intrinsic reward selector: `--intrinsic icm` instantiates the ICM model from P1.1, similar for `rnd` and `lpm`. `none` runs vanilla PPO with `r_intrinsic = 0`.
  - Logging: CSV row per env-step batch with columns `[step, ep_return, ep_len, visited_count_mean, r_ext, r_int, policy_loss, value_loss, intrinsic_loss]`.
  - Acceptance: script runs 10 000 env steps end-to-end on `noisy_tv × lpm` without crashing; CSV is non-empty.

- [ ] **P1.3 — Sanity sweep: 50 k env steps × 1 seed × all cells**
  - 3 variants × 4 intrinsic = 12 runs. Estimated ~30 min/run on Apple Silicon = ~6 h total wall-clock.
  - Driver: `experiments/phase1/run_sanity.sh`.
  - Deliverable: 12 CSV logs in `experiments/phase1/runs/sanity/`. Plot script `plot_visited.py` produces `visited_count vs env_step` figure with 4 lines (intrinsic methods) per variant subplot.
  - Acceptance: at 50 k steps, `LPM > {ICM, RND}` on `noisy_tv` and `action_noise` directionally (no statistical test yet).

- [ ] **P1.4 — Production runs: 500 k env steps × 3 seeds × all cells**
  - 12 cells × 3 seeds = 36 runs. Estimated ~5 h/run = ~7–8 days if serial. Plan: run two cells in parallel (~4 days), or cap at 250 k steps if 500 k is infeasible.
  - Deliverable: 36 CSV logs in `experiments/phase1/runs/main/`. Updated figure with mean ± 1 SD ribbons.
  - Acceptance: visual reproduction of the qualitative pattern in paper Figure 4-5 (LPM dominates on noisy variants, all methods comparable on `nonoise`). Compute paired t-test on final `visited_count` between LPM and the strongest baseline.

- [ ] **P1.5 — Phase 1 writeup**
  - File: `reports/phase1.md`. 2–3 pages.
  - Sections: setup, deviations from paper (smaller step budget, fewer seeds), main figure, statistical claim, limitations.
  - Commit: `phase1: reproduction writeup`.

---

## Phase 2 — Extend LPM to a sparse-reward task with structured stochasticity

**Setup.** `MiniGrid-Dynamic-Obstacles-8x8-v0` from the `minigrid` package. 4 balls perform a uniform random step each env step; collision = terminal + reward 0; reaching the green goal = `1 - 0.9 · (step / max_steps)`. Sparse, structured stochasticity from the moving balls.

**Hypotheses to test.**
- **H1 (control).** On `MiniGrid-DoorKey-8x8-v0` (sparse but deterministic), all four methods perform comparably — sparse reward is the dominant difficulty, intrinsic reward provides only marginal help.
- **H2 (curiosity trap).** On `Dynamic-Obstacles-8x8`, PPO+ICM and PPO+RND die more often than vanilla PPO. The moving balls act as a curiosity attractor; agents trained with these methods learn to approach them.
- **H3 (LPM is robust).** PPO+LPM's death rate is comparable to vanilla PPO. LPM identifies the balls' position as unlearnable and stops emitting bonus for them.

### Tasks

- [ ] **P2.1 — Install MiniGrid into the existing venv**
  - Command: `uv pip install --python LPM_exploration/.venv/bin/python minigrid`.
  - Verify: `gymnasium.make("MiniGrid-Dynamic-Obstacles-8x8-v0", render_mode="rgb_array").reset()` returns a valid obs.
  - Acceptance: a 10-line smoke script in `experiments/phase2/smoke.py` runs a random rollout and prints `obs.shape, action_space, reward range`.

- [ ] **P2.2 — Write the MiniGrid → CNN obs adapter**
  - File: `experiments/phase2/envs.py`.
  - MiniGrid's default obs is symbolic (7×7×3 tile-encoding). For parity with Phase 1's CNN, use `render_mode="rgb_array"` and wrap so `obs` is the RGB rendering resized to 84×84 (standard RL preprocessing).
  - Apply the same wrapper to `MiniGrid-DoorKey-8x8-v0` and `MiniGrid-Dynamic-Obstacles-8x8-v0`.
  - Acceptance: a 20-step random rollout returns 84×84×3 uint8 obs from both envs.

- [ ] **P2.3 — Adapt the Phase 1 training script for MiniGrid**
  - File: `experiments/phase2/train.py`. Mostly a copy of `experiments/phase1/train.py` with:
    - env builder uses `experiments/phase2/envs.py` instead of `miniworld_play.envs`.
    - new CLI: `--env {doorkey,dynobs} --lambda-intrinsic FLOAT` (mixes intrinsic into total reward: `r = r_ext + λ · r_int`, default λ = 0.1).
  - Acceptance: `train.py --env dynobs --intrinsic lpm --steps 5000` runs without error and the CSV shows non-zero ext rewards (occasional goal-reaches by chance).

- [ ] **P2.4 — Run H1 control (DoorKey-8x8)**
  - 4 intrinsic methods × 3 seeds = 12 runs, 500 k steps each.
  - Deliverable: `experiments/phase2/runs/h1_doorkey/`. Plot `ep_return vs env_step` with mean ± SD.
  - Acceptance: at 500 k steps all four methods reach `ep_return > 0.5`; differences statistically inconclusive (this is H1 — we *expect* no significant difference).

- [ ] **P2.5 — Run H2 + H3 main (Dynamic-Obstacles-8x8)**
  - 4 intrinsic methods × 3 seeds = 12 runs, 1 M steps each. Estimated ~10 h/run = ~5 days. Cap at 500 k if needed.
  - Tracked metrics per run: `ep_return`, `ep_length`, `death_rate` (= terminations not caused by reaching goal), `visited_cells_unique`.
  - Deliverable: `experiments/phase2/runs/h2h3_dynobs/`. Three figures:
    - ep_return convergence (4 methods, mean ± SD)
    - death rate over training
    - exploration coverage vs intrinsic-bonus magnitude (scatter)
  - Acceptance: PPO+ICM and/or PPO+RND show **significantly higher death rate** than PPO baseline (H2 supported); PPO+LPM's death rate is **not significantly higher** than PPO baseline (H3 supported). Use Welch's t-test on per-seed final-50k-step death rate.

- [ ] **P2.6 — Phase 2 writeup**
  - File: `reports/phase2.md`. 3–4 pages.
  - Sections: env description, hypothesis triple, main figures, statistical verdicts, caveats (small seed count, single env family, single ratio λ).
  - Commit: `phase2: extension writeup`.

---

## Phase 3 — Final report and presentation

- [ ] **P3.1 — Combined course report**
  - File: `reports/final.md` (or `.tex` if we end up wanting LaTeX).
  - Length: ~8 pages.
  - Sections: motivation, related work (LPM, ICM, RND, NGU, Disagreement), Phase 1 reproduction, Phase 2 extension, limitations, future work.
  - Cite the OpenReview concerns the project addresses (procedural generation, structured stochasticity, missing baselines).

- [ ] **P3.2 — Final presentation slides**
  - File: `reports/final_slides.{key,pdf}`.
  - 15-min talk. One figure per main claim. Live demo of `miniworld_play/play.py` if time allows.

---

## Risks and mitigations

| risk | likelihood | mitigation |
|---|---|---|
| Upstream LPM / RND / ICM model code is CUDA-only or has shape assumptions that block CPU port | medium | P1.1 catches this early; budget 1–2 days of porting buffer |
| Total training wall-clock exceeds available time | high | Cap at 250–500 k steps per run; reduce seeds from 3 to 2 if needed; skip P1.4 production sweep if P1.3 sanity already shows pattern |
| MiniGrid 84×84 RGB obs causes intrinsic-reward CNN to behave differently than on Miniworld 160×120 | low | Try MiniGrid's native 7×7×3 symbolic obs with a small MLP-based intrinsic model as fallback |
| Phase 2 hypothesis H3 fails (LPM also gets curiosity-trapped by moving balls) | medium | This is **still a publishable negative result** — write up as "LPM's robustness is limited to unstructured noise; structured stochasticity defeats it." Reviewer CKoM hinted at this. |
| `wandb` dependency in upstream code blocks running on machines without an account | low | Patch upstream `wandb.init` to a no-op when `WANDB_MODE=offline` or remove the call in our copy |

---

## Open decisions (settle before P1.2)

- **Logging backend.** wandb (upstream default, needs login) vs tensorboard vs raw CSV. Recommendation: **CSV** for reliability + a separate `plot_*.py` script. Adds no runtime dep.
- **PPO implementation source.** Upstream `LPM_exploration/Atari/exploration/algo/ppo.py` vs sb3 (`stable_baselines3 ≤ 1.8`) vs hand-written minimal PPO. Recommendation: **port the upstream PPO file** — it already integrates with the intrinsic-reward models we want to reuse.
- **Where do training scripts live.** Current proposal puts them at `experiments/phase{1,2}/`. Alternative: `LPM_exploration/Miniworld/` to keep "ours vs upstream" boundary clean. Recommendation: **`experiments/` top-level**, keeps `LPM_exploration/` an untouched snapshot.

---

## Status legend

`[ ]` not started · `[~]` in progress · `[x]` done.
Edit this file in place; commit each status flip.
