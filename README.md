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
├── materials/           — reference papers and course slides
├── reports/             — kickoff slides and project reports
└── LPM_exploration/     — embedded snapshot of the official LPM implementation
                          (upstream: github.com/Akuna23Matata/LPM_exploration,
                           see LPM_exploration/UPSTREAM.md for the commit pinned)
```

## Status

- **Noisy-MNIST toy experiment** (the paper's intro figure): reproduces qualitatively
  on Apple Silicon CPU in a few minutes. MSE-based intrinsic reward is fooled by
  stochasticity (Stoch ≈ 10× Det); LPM keeps Det > Stoch with bounded magnitude.
- **Miniworld 3D-maze exploration comparison** (`LPM_exploration/Miniworld/experiments/`):
  a CLI-driven A2C pipeline extracted from the upstream maze notebooks, comparing
  intrinsic-motivation methods (LPM / RND / ICM / MSE / none) across the three noise
  variants, with a new coverage-heatmap-evolution statistic. Runs downscaled on Apple
  Silicon CPU. See below.
- **Atari (Ms Pac-Man)** exploration comparison: see `LPM_exploration/Atari/experiments/`.
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

```bash
cd LPM_exploration/Miniworld/experiments

# run the grid (5 methods x 3 variants x 2 seeds = 30 runs, ~1.5h on CPU; resumable)
PYTHONPATH=. PYTORCH_ENABLE_MPS_FALLBACK=1 \
  ../../.venv/bin/python run_grid.py --steps 20000 --seeds 1 2 --device cpu

# aggregate -> table + coverage curves + beyond-wall / time-at-wall + heatmap evolution
PYTHONPATH=. ../../.venv/bin/python analyze.py    # writes figures/

# fast headless feasibility / throughput check, and the unit tests
PYTHONPATH=. ../../.venv/bin/python calibrate.py --variant noisy_tv --steps 300
PYTHONPATH=. ../../.venv/bin/python -m pytest tests/ -q
```

The **new statistic** is `fig_heatmap_evolution_<variant>_{density,frontier}.png`: a
rows=method × columns=training-window grid of top-down occupancy heatmaps showing how
each method's exploration spreads (or stalls at the noise wall) over time. Per-run CSVs,
position logs, and figures land in the gitignored `results/`, `positions/`, `figures/`.
The experiment design is written up in `latex_notes/2026-05-31-maze-exploration-design.tex`.
