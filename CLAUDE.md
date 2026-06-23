# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Workspace purpose

Lab project for the master-level course *Challenging Problems in Reinforcement Learning* (Lab Course SS 2026, AI & Formal Methods chair, Ruhr University Bochum). Group work (2–4 students). The project's chosen challenge topic is **Exploration vs. Exploitation**, and the work is built around reproducing and extending Hou, An, Du (2026), *Beyond Noisy-TVs: Noise-Robust Exploration via Learning Progress Monitoring* (arXiv:2509.25438).

Team: **Yingte** (designs + runs the experiments, owns the data) and **Youssef** (analysis + report). Youssef also maintains an external MiniGrid intrinsic-reward repo (`github.com/JosefGh/minigrid_intrinsic_reward`) referenced by the final-experiment design.

The workspace has moved past pure reproduction. Done so far: the Noisy-MNIST toy figure, and the **Miniworld 3D-maze exploration comparison** re-run at the paper's budget (paper-faithful log-space LPM, 6 methods × 3 noise variants × 64 seeds). That reproduction settled the headline noisy-TV robustness claim but also exposed that **coverage in an extrinsic-free maze does not separate methods** (a uniform-random policy covers the most), which is why the project's own contribution pivots to a **sparse-reward MiniGrid** setting where intrinsic motivation can actually pay off. That MiniGrid study (designed in `docs/SPEC.md`) is now **exercised and complete** — PPO on the 7×7 symbolic view across a difficulty ladder (DoorKey-5x5 / FourRooms / MultiRoom-N6), clean and noisy, comparing none / entropy / RND / LPM; written up in `latex_notes/2026-06-18-minigrid-intrinsic-exploration.{tex,pdf}` with data under `expr_data/minigrid/`. Atari (Ms Pac-Man) and Montezuma's Revenge remain not-yet-exercised.

## Top-level layout

- `README.md` — project description (public-facing).
- `CLAUDE.md` — this file.
- `LPM_exploration/` — **embedded snapshot** of the official LPM implementation. Upstream: `github.com/Akuna23Matata/LPM_exploration`; the pinned commit is recorded in `LPM_exploration/UPSTREAM.md`. **Treat this directory as third-party code we are reproducing** — do not refactor it casually. Any changes we make to upstream files should be noted in the "Local additions / deviations" section of `UPSTREAM.md` so reviewers can tell our changes from upstream.
- `docs/` — literature library + project design docs. Holds the reference PDFs (use citation-style filenames for new sources, e.g. `Aubret2019_IntrinsicMotivationSurvey.pdf`), the final-experiment spec, and the `superpowers/` design artifacts:
  - `Challenging Topics WS26.pdf` — the course's topic-introduction slide deck.
  - `Hou2026_LPM_BeyondNoisyTVs.pdf` (= `beyound-noisy-tvs.pdf`) — Hou, An, Du (UC Merced, ICLR 2026), *Beyond Noisy-TVs: Noise-Robust Exploration Via Learning Progress Monitoring* (arXiv:2509.25438). **The paper this project reproduces + extends.** Proposes LPM: intrinsic reward = improvement of the dynamics model, not prediction error → robust against the noisy-TV failure mode of RND/ICM. Code lives in `LPM_exploration/`. Envs studied: Noisy-MNIST, 3D maze (160×120 RGB), Atari.
  - `SPEC.md` — **the final-experiment design** (research questions, slide structure, work items): MiniGrid sparse-reward eval, a β intrinsic-coefficient sweep, and an LPM-vs-RND noise-robustness comparison.
  - `superpowers/specs/` and `superpowers/plans/` — design specs and plans from `superpowers:brainstorming` / `superpowers:writing-plans` sessions.
- `PROGRESS.md` — running log of team progress meetings (most recent first).
- `reports/` — slides and reports we author: `report1/` (kickoff Keynote/PowerPoint + asset-generation scripts), `report2/` (intrinsic-reward report, `.key` + `.pdf`), `report3/` (progress-meeting PDF).
- `expr_data/` — **canonical home for every experiment artifact** (results CSVs, position logs, figures, downloaded datasets). **Gitignored** — large/local-only. The harness scripts read & write here by default (e.g. `expr_data/miniworld/{results,positions,figures}`; the older 10-seed RAW-LPM snapshot is under `expr_data/miniworld/rawlpm_10seed/`; CIFAR-10 under `expr_data/datasets/`). See the SPEC requirement to keep raw data organized here with markdown explanations.
- `latex_notes/` — LaTeX write-ups of experiment designs/results (`*-maze-exploration-design`, `*-pacman-exploration-design`).
- `miniworld_play/` — **our** keyboard-controlled play tool for the paper's three Miniworld variants (`nonoise`, `noisy_tv`, `action_noise`). `envs.py` ports the upstream `MazeEnv` geometry (the 4-room hand-designed layout, the 25% sticky-action probability, the green-pixel→random-RGB transform on the noise wall) faithfully; `play.py` wraps it in a pygame window with first-person + top-down panes and writes a JSONL trajectory per session. Run via `./LPM_exploration/.venv/bin/python miniworld_play/play.py --variant noisy_tv`. Headless smoke test: same command with `--headless`.
- `LPM_exploration/.venv/` — uv-created Python 3.11 venv used for Noisy-MNIST, the Miniworld play tool, and the maze experiments. **Not committed** (gitignored). Includes: `torch`, `numpy`, `matplotlib`, `tqdm`, `python-mnist`, `jupyter`, `gymnasium`, `miniworld`, `pygame`, `Pillow`, `torchvision`, `pandas`, `pytest`.

## State of implementation

- **Noisy-MNIST toy experiment** runs end-to-end on CPU in the local uv venv and qualitatively matches the paper: MSE-based intrinsic reward is fooled by stochasticity (Stoch ≈ 10× Det); LPM keeps Det > Stoch with bounded magnitude. See `README.md` for the exact reproduction recipe.
- **Miniworld 3D-maze exploration comparison** (`LPM_exploration/Miniworld/experiments/`) is **exercised and complete.** A CLI A2C pipeline extracted from the upstream notebooks compares LPM / RND / ICM / MSE / none (+ a uniform-random control) across the three noise variants, at the paper's budget (Appendix C.2 config: λ=1, 50k steps, 64 seeds). Findings (see the two project memories + `UPSTREAM.md`): (1) the **noisy-TV robustness claim reproduces** — under `action_noise`, MSE fixates on the noise wall (TV-fixation ≈ 0.83, coverage collapses) while paper-faithful LPM stays robust (fixation ≈ 0.03); (2) the **"LPM explores more" claim does NOT reproduce** here — uniform-random covers the most, because this small extrinsic-free maze rewards randomness and training narrows the walk, so coverage is decoupled from noise-robustness; (3) LPM intrinsic-reward variance is large (std ≈ mean). LPM was made **paper-faithful** (log-space Eq 3, `reward_space="log"`, |D|=d gating; error-model lr 1e-3 not 1e-2). These findings motivate the MiniGrid pivot in `docs/SPEC.md`.
- **Atari (Ms Pac-Man) and Montezuma's Revenge** pipelines are **not yet exercised**. They depend on the legacy `gym` package (not `gymnasium`), a `stable_baselines3 ≤ 1.8` pin, ALE Atari ROMs (`AutoROM --accept-license`), and `wandb`. Full reproductions (~200M frames × 128 parallel envs for Montezuma, ~50M × 64 for Atari) are GPU-scale; the `Atari/experiments/` harness exists but no Atari data has been produced.
- **Sparse-reward MiniGrid study (`minigrid_exp/`, vendored+extended from Youssef's `minigrid_intrinsic_reward`): exercised and complete.** PPO (`ALGORITHM_NAME="ppo"`; DQN/UCB classical baseline retained but dormant in `ucb_dqn.py`) on the flattened 7×7 symbolic view, with a global observation-noise wrapper and RND/LPM intrinsic-reward wrappers, run via the chunked checkpoint→resume grid driver `run_grid.py` and aggregated by `analyze.py` / `make_report_figs.py`. Findings: (1) β is environment-dependent (usable ~0.001–0.005; 0.05 drowns the sparse reward); (2) intrinsic motivation is **difficulty-gated** — unneeded on easy/medium, decisive on hard MultiRoom-N6 (RND solves, baseline never reaches the goal); (3) under observation noise the trade-off **flips** — RND collapses while LPM stays robust (cleanest on DoorKey-5x5). All three envs **regenerate their layout every episode**, so eval returns average over the layout distribution. Write-up: `latex_notes/2026-06-18-minigrid-intrinsic-exploration.{tex,pdf}`; data/figures/`FINDINGS.md` under `expr_data/minigrid/`. The course report (slide deck) is still to be written.

## Chosen challenge topic: Exploration vs. Exploitation

Out of the five challenge topics presented in the slides (Safety, Sparse Rewards, Sim2Real, Explainability & Interpretability, Exploration vs. Exploitation), this project addresses **Exploration vs. Exploitation** — the fundamental RL trade-off between trying new actions and relying on the current best-known policy.

Slide-defined scope to anchor the project on:

- **Failure modes to motivate the work**: too much exploitation → sub-optimal policies, no adaptation; too much exploration → inefficient, risky, unstable.
- **Baseline / classical approaches** the project is expected to know and likely compare against: ε-greedy, Upper Confidence Bound (UCB), Thompson Sampling, softmax / Boltzmann exploration.
- **Open questions the slides flag as the interesting directions**:
  1. How to **incorporate prior knowledge** into exploration?
  2. How to **reduce risk** during exploration (overlap with the Safety topic)?
  3. How to **improve the efficiency** of exploration (sample-efficiency, directed exploration, intrinsic motivation)?
- **Reference cited on the slide**: Aubret, Matignon & Hassas (2019), *A survey on intrinsic motivation in reinforcement learning* (arXiv:1908.06976) — a natural starting point for the literature review. *(PDF not yet downloaded into `docs/`.)*

When the user asks about "the topic," "our problem," or similar without further qualification, assume they mean exploration vs. exploitation in the framing above.

## Communication style: math notation

**Never use LaTeX delimiters or display-math blocks** (`$...$`, `$$...$$`, `\frac`, `\sqrt`, `\sum` …). The user reads replies in a terminal where LaTeX source does **not** render — write all formulas as **inline linear notation** instead:

- `sqrt(x)` for square root, `x^2` for powers, `a / b` for fractions
- Subscripts written inline: `Q_t(a)`, `N_t(s,a)`
- Greek and operators as unicode characters: ε, γ, θ, Δ, Σ, π, ∞, ≤, ≥, ≈, →, ∈
- Sums / expectations / arg-ops inline: `sum_{a} f(a)`, `E[X]`, `argmax_a [...]`

Apply this to every formula in every reply in this workspace — explanations, derivations, pseudocode, notes saved into `docs/`. This is a hard rule, not a default.

## How to be useful here

- Use the `superpowers:brainstorming` skill when the user proposes a new experiment, ablation, or methodology — the project still has open scoping decisions even though some code is now running.
- Use `superpowers:writing-plans` once the scope of a multi-step change crystallizes.
- Distinguish carefully between *classical* exploration (ε-greedy, UCB, Thompson sampling, softmax — bandit and tabular roots) and *deep RL* exploration (count-based bonuses, RND, ICM, NoisyNets, bootstrapped DQN, Go-Explore, etc.) when discussing methods; the slides only enumerate the classical baselines, so the deep-RL extensions are likely where the project's novelty will live.
- **Compute:** the primary working dir `/data/yingte/projects/ChallengingRL-LPM` is a **128-core / ~1.1 TB-RAM Linux box (no GPU)** — it can run the full maze/MiniGrid grids via cross-run process parallelism (the 64-seed maze grid finished in ~35 min). The single-env A2C runs are batch-size-1, so use **1 thread per job and many processes** (`run_grid.py --jobs`). What's *not* feasible here is GPU-scale Atari/Montezuma (~50–200M frames); propose downscaled smoke tests for those. The user's local Apple Silicon Mac is the constrained machine, not this box.
- When touching code inside `LPM_exploration/`, log the change in `LPM_exploration/UPSTREAM.md` under "Local additions / deviations."
- Keep all experiment artifacts under `expr_data/` (gitignored) with markdown notes for observability, per the `docs/SPEC.md` requirements.

## Course logistics worth remembering

- Group formation: 2–4 students, Bachelor or Master.
- Topic selection is registered via a Moodle form ("Voting on Topic Preferences").
- The session following the topic-selection lecture begins a deep dive into sequential decision making — useful framing context, not a deliverable.
