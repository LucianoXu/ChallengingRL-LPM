# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Workspace purpose

Lab project for the master-level course *Challenging Problems in Reinforcement Learning* (Lab Course SS 2026, AI & Formal Methods chair, Ruhr University Bochum). Group work (2–4 students). The project's chosen challenge topic is **Exploration vs. Exploitation**, and the work is built around reproducing and extending Hou, An, Du (2026), *Beyond Noisy-TVs: Noise-Robust Exploration via Learning Progress Monitoring* (arXiv:2509.25438).

The workspace is in **early-reproduction phase**: the official LPM implementation has been embedded as a snapshot and the paper's Noisy-MNIST toy figure has been reproduced. The heavier experiments (Miniworld 3D maze, Atari, Montezuma's Revenge) and the project's eventual extension/ablation are still being scoped.

## Top-level layout

- `README.md` — project description (public-facing).
- `CLAUDE.md` — this file.
- `LPM_exploration/` — **embedded snapshot** of the official LPM implementation. Upstream: `github.com/Akuna23Matata/LPM_exploration`; the pinned commit is recorded in `LPM_exploration/UPSTREAM.md`. **Treat this directory as third-party code we are reproducing** — do not refactor it casually. Any changes we make to upstream files should be noted in the "Local additions / deviations" section of `UPSTREAM.md` so reviewers can tell our changes from upstream.
- `materials/` — literature library. PDFs of papers and the course slide deck. When adding new sources use citation-style filenames: `<FirstAuthor><Year>_<ShortTitle>.pdf` (e.g. `Aubret2019_IntrinsicMotivationSurvey.pdf` for the survey cited on the slides). Currently contains:
  - `Challenging Topics WS26.pdf` — the course's topic-introduction slide deck.
  - `Hou2026_LPM_BeyondNoisyTVs.pdf` — Hou, An, Du (UC Merced, ICLR 2026), *Beyond Noisy-TVs: Noise-Robust Exploration Via Learning Progress Monitoring* (arXiv:2509.25438). **The paper this project will reproduce + extend.** Proposes LPM: intrinsic reward = improvement of the dynamics model, not prediction error → robust against the noisy-TV failure mode of RND/ICM. Code lives in `LPM_exploration/`. Envs studied: Noisy-MNIST, 3D maze (160×120 RGB), Atari.
- `reports/` — kickoff slides and project reports we author. Currently contains `reports/report1/` with the kickoff Keynote/PowerPoint + the Python scripts that generated its assets.
- `miniworld_play/` — **our** keyboard-controlled play tool for the paper's three Miniworld variants (`nonoise`, `noisy_tv`, `action_noise`). `envs.py` ports the upstream `MazeEnv` geometry (the 4-room hand-designed layout, the 25% sticky-action probability, the green-pixel→random-RGB transform on the noise wall) faithfully; `play.py` wraps it in a pygame window with first-person + top-down panes and writes a JSONL trajectory per session. Run via `./LPM_exploration/.venv/bin/python miniworld_play/play.py --variant noisy_tv`. Headless smoke test: same command with `--headless`.
- `docs/superpowers/specs/` — design specs from `superpowers:brainstorming` sessions. Used when the user delegates a non-trivial implementation.
- `LPM_exploration/.venv/` — uv-created Python 3.11 venv used for Noisy-MNIST, the Miniworld play tool, and (eventually) the heavier experiments. **Not committed** (gitignored). Includes: `torch`, `numpy`, `matplotlib`, `tqdm`, `python-mnist`, `jupyter`, `gymnasium`, `miniworld`, `pygame`, `Pillow`, `torchvision`.

## State of implementation

- **Noisy-MNIST toy experiment** runs end-to-end on Apple Silicon CPU in the local uv venv and qualitatively matches the paper: MSE-based intrinsic reward is fooled by stochasticity (Stoch ≈ 10× Det); LPM keeps Det > Stoch with bounded magnitude. See `README.md` for the exact reproduction recipe.
- **Heavier experiments** (Miniworld hallways, Atari MsPacman with noisy-TV wrapper, Montezuma's Revenge) are **not yet exercised**. They depend on the legacy `gym` package (not `gymnasium`), a `stable_baselines3 ≤ 1.8` pin, ALE Atari ROMs (`AutoROM --accept-license`), and `wandb` (the upstream `main.py` calls `wandb.init` unconditionally). Full reproductions (~200M frames × 128 parallel envs for Montezuma, ~50M × 64 for Atari) are not feasible on the user's local Apple Silicon Mac and would need cluster/cloud compute.
- **What we still need to design / write:** the project's own extension or ablation (the novelty contribution), an evaluation plan, and the course report.

## Chosen challenge topic: Exploration vs. Exploitation

Out of the five challenge topics presented in the slides (Safety, Sparse Rewards, Sim2Real, Explainability & Interpretability, Exploration vs. Exploitation), this project addresses **Exploration vs. Exploitation** — the fundamental RL trade-off between trying new actions and relying on the current best-known policy.

Slide-defined scope to anchor the project on:

- **Failure modes to motivate the work**: too much exploitation → sub-optimal policies, no adaptation; too much exploration → inefficient, risky, unstable.
- **Baseline / classical approaches** the project is expected to know and likely compare against: ε-greedy, Upper Confidence Bound (UCB), Thompson Sampling, softmax / Boltzmann exploration.
- **Open questions the slides flag as the interesting directions**:
  1. How to **incorporate prior knowledge** into exploration?
  2. How to **reduce risk** during exploration (overlap with the Safety topic)?
  3. How to **improve the efficiency** of exploration (sample-efficiency, directed exploration, intrinsic motivation)?
- **Reference cited on the slide**: Aubret, Matignon & Hassas (2019), *A survey on intrinsic motivation in reinforcement learning* (arXiv:1908.06976) — a natural starting point for the literature review. *(PDF not yet downloaded into `materials/`.)*

When the user asks about "the topic," "our problem," or similar without further qualification, assume they mean exploration vs. exploitation in the framing above.

## Communication style: math notation

**Never use LaTeX delimiters or display-math blocks** (`$...$`, `$$...$$`, `\frac`, `\sqrt`, `\sum` …). The user reads replies in a terminal where LaTeX source does **not** render — write all formulas as **inline linear notation** instead:

- `sqrt(x)` for square root, `x^2` for powers, `a / b` for fractions
- Subscripts written inline: `Q_t(a)`, `N_t(s,a)`
- Greek and operators as unicode characters: ε, γ, θ, Δ, Σ, π, ∞, ≤, ≥, ≈, →, ∈
- Sums / expectations / arg-ops inline: `sum_{a} f(a)`, `E[X]`, `argmax_a [...]`

Apply this to every formula in every reply in this workspace — explanations, derivations, pseudocode, notes saved into `materials/`. This is a hard rule, not a default.

## How to be useful here

- Use the `superpowers:brainstorming` skill when the user proposes a new experiment, ablation, or methodology — the project still has open scoping decisions even though some code is now running.
- Use `superpowers:writing-plans` once the scope of a multi-step change crystallizes.
- Distinguish carefully between *classical* exploration (ε-greedy, UCB, Thompson sampling, softmax — bandit and tabular roots) and *deep RL* exploration (count-based bonuses, RND, ICM, NoisyNets, bootstrapped DQN, Go-Explore, etc.) when discussing methods; the slides only enumerate the classical baselines, so the deep-RL extensions are likely where the project's novelty will live.
- Before suggesting heavy commands (Atari/Montezuma training), remember the compute constraint above — propose downscaled smoke tests first.
- When touching code inside `LPM_exploration/`, log the change in `LPM_exploration/UPSTREAM.md` under "Local additions / deviations."

## Course logistics worth remembering

- Group formation: 2–4 students, Bachelor or Master.
- Topic selection is registered via a Moodle form ("Voting on Topic Preferences").
- The session following the topic-selection lecture begins a deep dive into sequential decision making — useful framing context, not a deliverable.
