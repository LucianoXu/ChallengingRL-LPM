# Upstream provenance

This directory is an **embedded snapshot** of the official LPM implementation
that accompanies the paper *Beyond Noisy-TVs: Noise-Robust Exploration via
Learning Progress Monitoring* (Hou, An, Du; UC Merced; ICLR 2026).

- **Upstream repository:** https://github.com/Akuna23Matata/LPM_exploration
- **Pinned commit:** `a295e3452491d9475c485a1a66a029eac4d0b55d`
- **Snapshot taken:** 2026-05-22

The upstream `.git/` directory was removed when embedding — these files are
tracked as part of the parent `ChallengingRL` repository, not as a submodule.

To diff against upstream or pull newer changes:

```bash
git clone https://github.com/Akuna23Matata/LPM_exploration.git /tmp/lpm_upstream
diff -ruN /tmp/lpm_upstream LPM_exploration
```

## Local additions / deviations

Any files we modify or add inside `LPM_exploration/` (e.g. fixes, configs,
new ablation scripts) should be noted here so reviewers can tell what is
ours vs. what is upstream.

- **2026-05-22:** The Miniworld `MazeEnv` geometry (the 4-room layout,
  agent spawn, action semantics, sticky-action probability, and
  green-pixel→random RGB transform) has been ported into
  `../miniworld_play/envs.py` so a human can play the three variants by
  keyboard. The upstream notebooks were left untouched; the port is
  faithful, including the upstream `R<800` mask quirk (vestigial since
  uint8 max is 255). The `nonoise` upstream notebook depends on a missing
  `n_shape.py` module (not in upstream), so its `NoNoiseEnv` was
  synthesised from the geometry shared with the other two variants.

- **2026-05-31 — Atari LPM smoke-test fixes.** The committed `Atari/` snapshot
  does **not** run as-is; the following minimal edits were needed just to get a
  single-seed `--algo ppo-improvement` (LPM) run on `MsPacmanNoFrameskip-v4` to
  reach the training loop. All are marked inline with `# local fix:` comments.
  1. `Atari/exploration/models/__init__.py` — commented out `from .lbs import LBS`
     and `from .tdd import ...` (both modules absent from the snapshot; unused by
     the LPM path).
  2. `Atari/main.py` — wrapped `import exploration.environments` (subpackage
     absent; MountainCar/Ant only) and the `tdd2`/`eme` model imports (modules
     absent) in `try/except ModuleNotFoundError`.
  3. `Atari/exploration/envs.py` — added `import ale_py; gym.register_envs(ale_py)`
     at module top, else ALE envs are unregistered inside `SubprocVecEnv` workers
     (`NameNotFound: MsPacmanNoFrameskip`).
  4. `Atari/exploration/envs.py:_thunk` — fixed an unbound-`obs_shape` `NameError`
     on the live path (replaced with `env.observation_space.shape`).
  5. `Atari/exploration/envs.py:ProcessFrame84` — output made channel-first
     `(1,84,84)` instead of HWC `(84,84,1)` so the 4-frame stack is `(4,84,84)`
     as the CNN expects (was crashing with "Input 84x1 too small").
  NOT yet fixed (these block a *faithful* reproduction but not the timing run):
  `main.py:137` has `int_coeff = 0.  # test`, which zeroes the intrinsic reward
  for the entire LPM/curiosity family (run reduces to extrinsic-only PPO);
  `improve.py:228` has the `/255` pixel normalization commented out;
  `run.sh` launches `--algo eme` (not LPM) on the clean variant with a hardcoded
  `device='cuda'`. See the project notes / analysis for the full list.

- **2026-05-31 — `Atari/play_lpm.py` (our addition).** A playback/visualization tool
  (not upstream) to watch a trained `ppo-improvement` agent play. Loads a saved
  policy checkpoint, reuses the repo's own `ProcessFrame84` + SB3 wrappers + the
  checkpoint's obs mean/std so preprocessing matches training, and either records a
  GIF/MP4 or shows a live pygame window (`--live`). Supports the `--noisy` CIFAR
  variant. Trained checkpoints live under the gitignored `Atari/trained_models/`.

- **2026-05-31 — Atari RND/ICM/AMA smoke-test fixes (exploration-baseline comparison).**
  Goal: get `--algo {rnd,icm,ama} --beta 1` to reach the training loop and log sane
  episode scores on `MsPacmanNoFrameskip-v4`. Findings:
  1. `Atari/exploration/models/RND.py` — runs **unmodified**. Random target + predictor
     (no decoder/Sigmoid, no `/255`), uses the passed `device`. Smoke: exit 0, sane
     `ep_score_mean` ~197–269.
  2. `Atari/exploration/models/icm.py` — runs **unmodified**. Forward+inverse models on
     signed normalized features, no decoder/Sigmoid/`/255`, uses `device`. Smoke: exit 0,
     `ep_score_mean` ~205–376.
  3. `Atari/exploration/models/ama.py` — runs **unmodified**. The pix2pix UNet decoder
     uses Tanh on intermediate deconvs but **no activation on the final `deconv3`**, so the
     reconstruction `mu` is unbounded — correct for the signed N(0,1) target (no Sigmoid bug
     unlike LPM's improve.py). Uses `device`. Smoke: exit 0, `ep_score_mean` ~215–356. (Raw
     intrinsic reward `mse - predicted_variance` is large/signed by construction; episode
     scores are unaffected since ext_coeff=1.)
  4. `Atari/main.py` (ama branch, ~L199) — replaced the `if args.use_dones: ext=1,int=1e-3
     else ext=0,int=1` coefficient block with `ext_coeff=1.; int_coeff=args.beta` (marked
     `# local fix:`) so AMA uses `r = ext + beta*int` consistently with the rnd/icm branch
     for a fair comparison. (The default `--use-dones` False path had set `ext_coeff=0.`,
     i.e. pure-intrinsic — episodes would still log but the policy ignored game score.)

- **2026-05-31 — `Atari/pacman_play.py` (our addition).** Human keyboard player for
  Ms Pac-Man (pygame), same spirit as `../miniworld_play/play.py`. Arrows/WASD move
  (diagonals supported), R restart, Q/Esc quit; `--noisy` exposes the CIFAR idle
  actions for a human to trigger; `--headless` runs a no-window self-test.

- **2026-05-31 — `Miniworld/experiments/` (our addition): the maze exploration
  comparison.** The upstream Miniworld experiment exists only as three Jupyter
  notebooks (`miniworld_hallway_{nonoise,with_noisyTV,with_action_based_noise}.ipynb`)
  with an embedded **A2C** (RMSprop, GAE γ=0.99/λ=0.95, single-epoch update every 64
  steps), single environment, 50k steps/run, no CLI. We extracted that engine into an
  importable, CLI-driven package to compare intrinsic-motivation methods across the
  three noise variants and to add a per-step position log for coverage heatmaps. The
  notebooks were left untouched. Specifics:
  1. `models.py` — `CNNFeatureExtractor`, the decoder, the LPM prediction +
     uncertainty nets, and the LPM reward `clip(eta*E[err] - err, <=0.5)` (eta=1.0)
     are lifted **verbatim** from the notebooks, parameterised by `device` instead of
     a module global. `MSEModel` is the notebook's decoder next-state-prediction
     curiosity (the pure noisy-TV victim).
  2. **New methods not in the notebooks:** `ICMModel` (canonical inverse+forward
     dynamics on the shared encoder — the notebook's "CuriosityModel" is *not* real
     ICM, just a decoder) and `RNDModel` (frozen random target + trainable predictor).
     Both are built on the same `CNNFeatureExtractor` for comparability.
  3. **Uniform reward combination.** The notebooks combine rewards with inconsistent
     ad-hoc offsets (`-0.002`, `-0.005`, `min(·,1.0)`) yet a single `lambda=0.1`
     despite very different reward scales. For a *fair* comparison `a2c.py` instead
     applies **running mean/std normalisation** (Welford, RND-style) to every method's
     intrinsic reward, then `combined = lambda_intrinsic * normalise(r_int)`
     (extrinsic is 0 in this env). Toggle via `A2CAgent(normalize_intrinsic=...)`.
  4. **Env reused, not re-ported:** `maze_envs.make_env` imports the three variant
     classes from `../../miniworld_play/envs.py` (single source of truth for geometry).
  5. **Downscaling for local compute:** runs use `--steps 20000` (vs upstream 50k) and
     default to CPU (`--device cpu`) because the decoder methods (LPM/MSE) are faster
     on CPU than MPS, where `ConvTranspose2d` falls back off-device. `--obs-scale` can
     reduce the 160×120 render resolution if needed (faithful default is 1.0;
     `action_noise` requires scale 1.0 since its CIFAR resize targets 160×120).
  Run with `run_grid.py` → per-run CSV + position `.npz` → `analyze.py` produces the
  coverage table, coverage curves, the beyond-wall / time-at-wall figures, and the
  coverage-heatmap-evolution figures. Artifacts under `results/`, `positions/`,
  `figures/` are gitignored.
