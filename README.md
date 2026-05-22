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
- **Miniworld / Atari / Montezuma** pipelines: not yet exercised. They depend on legacy
  `gym` (not `gymnasium`) + `stable_baselines3 ≤ 1.8` + ALE ROMs; large runs need
  GPU compute that isn't available locally.

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
