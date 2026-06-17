# ChallengingRL — Course Lab Project on Exploration vs. Exploitation

Lab project for *Challenging Problems in Reinforcement Learning* (Lab Course SS 2026,
AI & Formal Methods chair, Ruhr University Bochum). The chosen challenge topic is
**Exploration vs. Exploitation**, and the project is built around reproducing and
extending:

> Hou, An, Du (UC Merced, ICLR 2026), *Beyond Noisy-TVs: Noise-Robust Exploration
> via Learning Progress Monitoring*. arXiv:2509.25438.

## Repo layout

```
ChallengingRL/
├── CLAUDE.md            — guidance for Claude Code (working agreements, scope, notation)
├── PROGRESS.md          — running log of team progress meetings
├── docs/                — reference papers, course slides, and the final-experiment SPEC.md
├── reports/             — slides and reports we author (report1 kickoff, report2 intrinsic-reward, report3 progress)
├── latex_notes/         — LaTeX write-ups of the experiment designs/results
├── expr_data/           — canonical home for all experiment artifacts + datasets (gitignored)
├── miniworld_play/      — keyboard-controlled play tool for the three Miniworld noise variants
└── LPM_exploration/     — embedded snapshot of the official LPM implementation
                          (upstream: github.com/Akuna23Matata/LPM_exploration,
                           see LPM_exploration/UPSTREAM.md for the commit pinned)
```

## Status

- **Noisy-MNIST toy experiment** (the paper's intro figure): reproduces qualitatively
  on CPU in a few minutes. MSE-based intrinsic reward is fooled by stochasticity
  (Stoch ≈ 10× Det); LPM keeps Det > Stoch with bounded magnitude.
- **Miniworld 3D-maze exploration comparison** (`LPM_exploration/Miniworld/experiments/`):
  **exercised and complete.** A CLI-driven A2C pipeline extracted from the upstream maze
  notebooks compares intrinsic-motivation methods (LPM / RND / ICM / MSE / none + a
  uniform-random control) across the three noise variants, with a coverage-heatmap-evolution
  statistic. Re-run at the paper's Appendix-C.2 budget (λ=1, 50k steps, 64 seeds) with a
  **paper-faithful log-space LPM**. Key results: the **noisy-TV robustness claim reproduces**
  (under `action_noise`, MSE fixates on the noise wall while LPM stays robust), but the
  **"LPM explores more" claim does not reproduce** — in this small, extrinsic-free maze a
  uniform-random policy covers the most, so coverage turns out to be decoupled from
  noise-robustness. This is what motivates the move to a sparse-reward setting (below). See below.
- **Final experiment (designed, see `docs/SPEC.md`):** a sparse-reward **MiniGrid** study — a
  β intrinsic-coefficient sweep and an LPM-vs-RND noise-robustness comparison, where intrinsic
  motivation should actually help (unlike the coverage-saturated maze).
- **Atari (Ms Pac-Man)** exploration comparison: harness in `LPM_exploration/Atari/experiments/`;
  not yet exercised (no data produced).
- **Montezuma** full-scale pipeline: not yet exercised; large runs need cluster/cloud GPU.

## Reproducing the Noisy-MNIST smoke test

```bash
cd LPM_exploration
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python torch numpy matplotlib tqdm python-mnist jupyter nbconvert ipykernel

# fetch raw MNIST (gitignored)
mkdir -p Noisy_MNIST/data && cd Noisy_MNIST/data
for f in train-images-idx3-ubyte.gz train-labels-idx1-ubyte.gz \
         t10k-images-idx3-ubyte.gz   t10k-labels-idx1-ubyte.gz; do
  curl -sS -O "https://storage.googleapis.com/cvdf-datasets/mnist/$f"
done
gunzip -f *.gz && cd ../..

# run the notebook end-to-end
./.venv/bin/jupyter nbconvert --to notebook --execute \
  Noisy_MNIST/mnist_curiosity.ipynb \
  --output mnist_curiosity_executed.ipynb \
  --ExecutePreprocessor.timeout=1800
```

## Playing the Miniworld variants by hand

`miniworld_play/` is our keyboard-controlled tool for the paper's three Miniworld
scenarios (faithful port of the upstream 4-room maze: 25% sticky actions, green-pixel
→ random-RGB transform on the noise wall). Useful for sanity-checking geometry and
seeing what the noisy-TV failure mode actually looks like from the agent's POV.

```bash
# headline noisy-TV scenario (sticky actions + random-RGB noise wall)
./LPM_exploration/.venv/bin/python miniworld_play/play.py --variant noisy_tv

# baseline without noise or sticky actions
./LPM_exploration/.venv/bin/python miniworld_play/play.py --variant nonoise

# action-conditioned noise (press N to "look at the noisy TV")
./LPM_exploration/.venv/bin/python miniworld_play/play.py --variant action_noise

# headless smoke test: 50 random steps, save a screenshot, exit
./LPM_exploration/.venv/bin/python miniworld_play/play.py --variant noisy_tv --headless
```

Keys: arrows / WASD to move, N for the noisy-TV action, R reset, T toggle stickiness,
M toggle side panel, SPACE pause, F12 screenshot, Q / ESC quit. Each session writes
a JSONL trajectory to `miniworld_play/recordings/`.

## Reproducing the 3D-maze exploration comparison

`LPM_exploration/Miniworld/experiments/` extracts the maze notebooks' A2C engine into a
CLI trainer and compares exploration methods on coverage. The maze has **no extrinsic
reward**, so "exploration ability" is pure spatial coverage — and the noisy-TV failure
mode is geometric: the random-RGB noise wall at z≈8 separates the lower rooms from the
far room4, so a method fooled by noise lingers at the wall while LPM should push past it.

All artifacts read/write under `<repo>/expr_data/miniworld/{results,positions,figures}`
(gitignored) by default — runs are resumable (a finished run's `positions/*.npz` is the
completion marker). On the 128-core box, parallelise with `--jobs`.

```bash
cd LPM_exploration/Miniworld/experiments

# paper-faithful grid (6 methods incl. the uniform-random control x 3 variants x 64 seeds,
# 50k steps, Appendix-C.2 config; ~35 min at --jobs 100 on the 128-core box; resumable)
PYTHONPATH=. ../../.venv/bin/python run_grid.py \
  --methods lpm rnd icm mse none uniform \
  --steps 50000 --seeds $(seq 1 64) --device cpu --jobs 100

# aggregate -> table + coverage curves + beyond-wall / time-at-wall + heatmap evolution
# (defaults to expr_data/miniworld/{results,positions} -> expr_data/miniworld/figures)
PYTHONPATH=. ../../.venv/bin/python analyze.py

# fast feasibility / throughput check, and the unit tests
PYTHONPATH=. ../../.venv/bin/python calibrate.py --variant noisy_tv --steps 300
PYTHONPATH=. ../../.venv/bin/python -m pytest tests/ -q
```

LPM defaults to the **paper-faithful log-space reward** (Eq 3, `LPMModel(reward_space="log")`
in `models.py`); the older raw-MSE form is still selectable in code via `reward_space="raw"`
(its 10-seed snapshot is archived under `expr_data/miniworld/rawlpm_10seed/`). The heatmap statistic is
`fig_heatmap_evolution_<variant>_{density,frontier}.png`: a rows=method × columns=training-window
grid of top-down occupancy heatmaps showing how each method's exploration spreads (or stalls
at the noise wall) over time. The experiment design is written up in
`latex_notes/2026-05-31-maze-exploration-design.tex`.
