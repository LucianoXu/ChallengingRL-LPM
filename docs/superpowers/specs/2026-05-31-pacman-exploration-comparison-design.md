# Design: Exploration-Algorithm Comparison on Ms Pac-Man

- **Date:** 2026-05-31
- **Status:** Approved (brainstorming) → next: implementation plan (writing-plans)
- **Course context:** *Challenging Problems in RL* (SS 2026), topic **Exploration vs. Exploitation**, built around reproducing + extending Hou, An, Du (2026) *Beyond Noisy-TVs: LPM* (arXiv:2509.25438).

## 1. Motivation & headline

Compare several exploration algorithms (classical + deep intrinsic-motivation, **including LPM**) on Atari **Ms Pac-Man**, measure their relative strength, and **explain why** they differ. Two intertwined questions:

- **RQ1 — Noise robustness (backbone, reproduction).** Under an injected noisy-TV (action-triggered CIFAR noise), which methods collapse and which survive? How large is LPM's advantage over prediction-error curiosity (RND/ICM) and over the noise-robust curiosity baseline (AMA)?
- **RQ2 — Stochasticity attribution (novelty, the user's hypothesis).** LPM appears to win even on **clean** Ms Pac-Man. Hypothesis: the game's intrinsic randomness (random ghost movement) acts as aleatoric noise that fools prediction-error methods, giving LPM an edge even without injected noise. Because the ghost AI is fixed in the ROM and cannot be tuned, we probe the **general claim** — "LPM's advantage grows with environment stochasticity" — using ALE **sticky actions** (`repeat_action_probability`) as a controllable stochasticity knob.

The combined story: a defensible robustness reproduction (RQ1) **plus** an original mechanistic analysis (RQ2).

## 2. Methods (6 trained, all on a shared PPO base)

All methods share the same PPO base algorithm so that differences are attributable to the **exploration mechanism**, not the base learner. Intrinsic-motivation methods plug in via the repo's curiosity-model interface (`--algo`), all with intrinsic weight `int_coeff = beta` (default 1, the fix from commit 264c65a).

| Class | Method | Mechanism | Code (`--algo`) |
|---|---|---|---|
| Learning progress | **LPM** | reward = improvement of dynamics model | `ppo-improvement` (already fixed/runs) |
| Prediction error | **RND** | error vs fixed random target net | `rnd` |
| Prediction error (controllable features) | **ICM** | inverse-dynamics feature prediction error | `icm` |
| Noise-robust curiosity | **AMA** | aleatoric-uncertainty–filtered curiosity | `ama` |
| Classical: softmax / Boltzmann | **plain PPO** | entropy-regularized stochastic policy, no intrinsic reward | `ppo` |
| Classical: ε-greedy | **ε-greedy-PPO** | PPO policy + ε-random action layer | `ppo` + **new wrapper** |

**Discussion-only (not trained):** UCB, Thompson sampling — no scalable deep-RL drop-in (UCB needs counts/uncertainty, Thompson needs posterior sampling / bootstrapped ensembles). The report argues that count-based bonuses / LPM are the *scalable descendants* of these — this directly engages the course's open questions.

**Note on classical exploration in a policy-gradient setting:** plain PPO's entropy-regularized action sampling *is* softmax/Boltzmann exploration; ε-greedy is added as a thin action wrapper. UCB/Thompson remain conceptual.

## 3. Experimental grid (controlled, not full-factorial)

Two manipulation axes, kept separate to bound run count.

**RQ1 — noise axis (CIFAR action-noise):**
- 6 methods × {clean, noisy} × 3 seeds = **36 runs**
- `clean` = stock Ms Pac-Man; `noisy` = action-noise variant (`--noisy --randop 2`: idle actions replace the frame with a random CIFAR-10 image).

**RQ2 — stochasticity sweep (clean only, sticky actions):**
- `repeat_action_probability` ∈ {0.0, 0.25, 0.5}; the `0.0` point **reuses** the RQ1 clean runs (not re-run).
- Sweep only the 4 most diagnostic methods: **LPM, RND, ICM, plain-PPO** × {0.25, 0.5} × 3 seeds = **24 runs**
- (AMA deliberately excluded from the sweep to save effort; included in RQ1.)

**Totals:** ≈ **60 runs × ~20 min on MPS ≈ 20 h compute** (2–3 overnight batches). Compute is *not* the bottleneck — **getting the other 5 baselines to run is** (each likely has latent bugs like LPM did).

**Fixed training config (per run):** PPO, 1M aggregate frames, 16 processes, num-steps 128, ppo-epoch 3, num-mini-batch 8, lr 1e-4, clip 0.1, entropy-coef 0.001, γ 0.99, GAE; MPS device. (Matches the paper's Atari hyperparameters / commit 264c65a.)

**Seeds:** 3 (enough for mean ± std; acknowledged as small-sample).

## 4. Metrics

- **Primary:** final extrinsic game score = mean episode return over the last ~10% of updates, reported mean ± std over seeds, per (method, condition).
- **RQ1 headline — noise-robustness drop:** `(score_clean − score_noisy) / score_clean` per method. Prediction: LPM (and AMA) smallest; RND/ICM largest. Reproduces the paper's central result.
- **RQ2 headline — stochasticity-advantage curve:** `(score_LPM − score_RND)` and `(score_LPM − score_ICM)` as a function of `repeat_action_probability`. Prediction: increasing → supports the hypothesis.
- **Attribution (auxiliary):** learning curves (score vs frames, clean & noisy); **intrinsic-reward trajectories** per method (LPM should decay toward 0 as the world model converges; RND/ICM under noise should stay elevated — direct evidence of being "stuck on noise").

## 5. Infrastructure to build (scope of the implementation plan)

1. **Get all 5 remaining baselines running** (RND, ICM, AMA, plain-PPO, ε-greedy-PPO) — Phase 1, the largest unknown; debug each as LPM was debugged (missing files, shapes, device, Sigmoid/normalization, `int_coeff`).
2. **Clean per-run logging:** write a CSV per run (`update, frames, ep_score, ep_score_std, int_rew, ext_rew, pred_loss, unc_loss, fps`) instead of grep-parsing stdout (last time the episode-score line was filtered out). Add this to `main.py`'s logging path.
3. **ε-greedy-PPO wrapper** (~30 lines): with prob ε take a uniform-random action over the base action set; otherwise the PPO action. ε schedule configurable. Verify plain-PPO is the softmax baseline.
4. **Sticky-action config:** pass `repeat_action_probability` through env creation (env kwarg or v0/v4 selection).
5. **Grid runner + analysis:** a script that launches (method × condition × seed) runs (sequential on MPS), and a parser/plotter that produces the results table + 3 figures (clean/noisy learning curves, noise-robustness bar chart, stochasticity-sweep curve) + the intrinsic-reward analysis.

## 6. Deliverables (for the course report)

- Main results table: methods × conditions (final score, clean→noisy drop).
- 3 figures: learning curves; noise-robustness bar chart; stochasticity-sweep curve.
- Intrinsic-reward attribution analysis.
- Written analysis: why each method wins/loses; the RQ2 stochasticity story and its limits.

## 7. Risks & honest caveats

- **Baselines may each be broken.** Phase 1 may overrun; if a method cannot be made to run, downgrade it to "attempted, not working," document it (itself a finding), and proceed with the rest.
- **Sticky actions ≠ ghost randomness.** It is a controllable *proxy* for aleatoric stochasticity, not an isolation of ghost behavior. The report must state this: we test "stochasticity → LPM edge" generally, with ghost-randomness as qualitative motivation, not a clean isolation.
- **Short budget / small sample.** 1M frames × 3 seeds yields *qualitative trends*, not rigorous statistical significance. Report accordingly.
- **Episodic-bonus family absent.** EME and TDD/EDT model files are missing from the snapshot, so the comparison covers the curiosity + learning-progress + classical families, not episodic bonuses. Noted as a scope limit.

## 8. Out of scope (YAGNI)

- Matching SOTA Ms Pac-Man scores (MuZero/Agent57 territory) — not the goal.
- Montezuma's Revenge / other games.
- A separate value-based (DQN) agent — would confound exploration strategy with base algorithm.
- RAM-based ghost-position heatmaps (Approach 2) — dropped in favor of the sticky-action sweep (Approach 1).
- UCB/Thompson implementations — discussion-only.

## 9. Resolved decisions

- Headline: **combined** (RQ1 robustness backbone + RQ2 stochasticity attribution).
- Method tier: **Core** (LPM, RND, ICM, AMA) + classical (plain-PPO/softmax, ε-greedy-PPO); UCB/Thompson discussion-only.
- RQ2 probe: **Approach 1** (sticky-action stochasticity sweep).
- Seeds: **3**. Sweep subset: **LPM, RND, ICM, plain-PPO** (AMA excluded from sweep).
