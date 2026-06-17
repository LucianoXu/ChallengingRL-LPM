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

- **2026-06-02 — paper-faithful maze rerun on a 128-core Linux box (50k × 10 seeds).**
  The original maze figures (`*_mac_20k/`, now backed up) were `--steps 20000 × 2 seeds`
  on the user's Apple-Silicon Mac and did not separate the methods (NONE competitive
  everywhere, between-seed std ≈ mean). Diagnosis: underpowered, not a code bug — the
  `action_noise` noisy-TV result already proved the mechanism is correct. We re-ran the
  full grid at the paper's actual maze budget (Fig.3 / §5.2: **50,000 exploration steps,
  averaged over 10 seeds**) using the box's parallelism. Setup notes:
  1. **New Linux venv** at `LPM_exploration/.venv` (uv, CPython 3.11): torch 2.12+cpu,
     torchvision 0.27+cpu, gymnasium 1.3.0, miniworld 2.1.0, pyglet 1.5.31, pandas,
     matplotlib. The Mac venv is gitignored / absent here.
  2. **Headless render, no GPU:** set `PYGLET_HEADLESS=true` (pyglet EGL + Mesa
     `libEGL_mesa` software GL). No DISPLAY / no xvfb needed. Also pin `LP_NUM_THREADS=1`
     so llvmpipe doesn't oversubscribe when many workers run.
  3. **Parallelism:** `run_grid.py --jobs 120 --threads-per-job 1` (5×3×10 = 150
     independent runs). 1 thread/job is optimal — 2 threads only gave 1.25× per run, so
     max-processes wins. Measured single-thread FPS: LPM ≈ 41, NONE ≈ 112, action_noise
     ≈ 30. Makespan ≈ 35 min at ~92% user CPU, 0 failures, ~140 GB RAM peak (of 1.1 TB).
  4. **CIFAR pre-staged:** download CIFAR-10 once before launch, else the 50 parallel
     `action_noise` workers race on the same `download=True` target and corrupt it.
  5. **Resume gotcha:** `run_grid.is_complete` keys on the position `.npz` existing, not
     on step count, so the old 20k `positions/` were moved aside (→ `positions_mac_20k/`)
     before the 50k rerun or they'd be skipped as "done".
  Result (`figures/table_*.csv`): the **noisy-TV claim reproduces cleanly** — under
  `action_noise`, MSE is trapped (coverage 0.024, TV-fixation 0.983±0.023) while LPM is
  the best intrinsic method (coverage 0.623, TV-fixation 0.048±0.075, below uniform 0.2).
  Two non-budget issues persist (config/geometry, not step count): (a) the no-intrinsic
  NONE baseline stays competitive on raw coverage (0.62–0.70) because upstream
  `entropy_coef=0.05` + per-batch return standardisation keep every policy near a uniform
  random walk, which already covers this small 4-room maze well (the paper has no NONE
  baseline); (b) the `noisy_tv` pixel-wall is a passable place, not an arresting action,
  so it is a weak trap in this geometry and methods don't separate there.

- **2026-06-02 (later) — aligned hyperparameters to the paper's Appendix C.2, not the
  notebook.** The "NONE competitive / intrinsic signal doesn't separate coverage" finding
  above traced to a hyperparameter mismatch: our defaults were taken from the upstream
  *notebooks*, which **disagree with the paper's own Appendix C.2 (MiniWorld)**. C.2
  specifies, for all methods: **intrinsic weight λ=1** (notebook used 0.1 → 10x too weak),
  **entropy coefficient 0.03** (notebook used 0.05), and the combined reward is **raw
  `extrinsic + λ·r_intrinsic`** with no running-std normalization (we had added Welford
  normalization "for fairness"). Architecture, optimizer, LRs, γ, value/grad-clip,
  update-freq, dynamics-LR (1e-3), and error-buffer (100) already matched C.2. We changed
  the `a2c.py` / `train_maze.py` defaults to C.2 (`lambda_intrinsic=1.0`, `entropy_coef=0.03`,
  `normalize_intrinsic=False`) and exposed `--entropy-coef` / `--normalize-intrinsic` flags
  so the notebook config remains reproducible. Smoke check (LPM, `action_noise`): the early
  `value_loss` spike is a transient that settles to ~1.0 by ~12 updates (same as before);
  crucially the policy is now **directed** — action distribution `[0.06,0.15,0.72,0.06,0.008]`
  (72% move-forward, 0.8% noisy-TV) with intermediate entropy, no longer pinned at ln5. The
  50k×10-seed grid was re-run under C.2; the notebook-config results are backed up under
  `*_notebook_50k/`. NB: λ=1 raw is faithful for the paper's own methods (LPM, MSE-decoder);
  our added RND/ICM have different raw scales and may warrant separate handling for a strictly
  fair cross-method coverage comparison.

- **2026-06-02 (final) — uniform-random control, seed-averaged heatmaps, cleanup.**
  1. `train_maze.py --random-policy` added: bypasses the A2C and samples actions uniformly at
     random every step with no learning (skips reward/memory/updates), reusing the same
     coverage/CSV/npz logging. It is the proper "does any of this beat chance?" control, distinct
     from `none` (whose A2C policy is still shaped by entropy + value bootstrap). Result: the uniform
     control covers the **most** in all three variants (nonoise 0.515, noisy_tv 0.688, action_noise
     0.725), lowest variance, beating every trained policy incl. LPM — in this small extrinsic-free
     maze coverage rewards randomness and any learned policy commits to a narrower walk. Uniform
     TV-action share = 0.200 (the 1/5 sanity check). Noise robustness (LPM 0.4% vs MSE 87% fixation)
     is unaffected and decoupled from coverage.
  2. `heatmaps.py` / `analyze.py`: coverage-heatmap-evolution figures now **average over all seeds**
     (density = mean per-cell step-count; frontier = fraction of seeds that ever visited each cell,
     a 0-1 visit-probability map) instead of using seed 1.
  3. Cleanup: the older backup result dirs (`*_mac_20k/`, `*_notebook_50k/`) were **deleted** — only
     the C.2 results (`results/`, `positions/`, `figures/`, now incl. `uniform-*`) are kept. The
     `latex_notes` design note was rewritten as a single C.2-only document.

- **2026-06-02 (LPM made paper-faithful + 64-seed rerun).** Acting on the code-vs-paper
  audit, `Miniworld/experiments/models.py:LPMModel` was corrected from the notebook's raw-space
  reward to the paper's Eq (1)-(3) + Algorithm 1 (default `reward_space="log"`):
  1. **Log-space reward (Eq 1/3):** `r = g_phi(s,a) − log(MSE)` (a difference of log-errors),
     replacing the notebook's `min(0.5, eta·exp(g_phi) − MSE)`. No eta, no 0.5 clip.
  2. **|D|=d gating (Alg 1 L6):** reward is 0 until the error queue fills (buffer_size=100).
  3. **g_phi updated every dynamics update** (`update_unc_every=1`, Alg 1 L9-11; was every 5th).
  4. **Error-model lr 1e-3, not 1e-2.** Root cause found via an overfit probe: under the
     log-space objective the notebook's 1e-2 drives g_phi into the [-10,10] clamp (zero gradient
     → dead error model → reward pinned at ≈ −6); 1e-3 fits the log-error targets cleanly (probe
     loss 39.8 → 0.001) and is the only lr C.2 specifies. After the fix, an end-to-end smoke shows
     unc_loss 11.7 → ~0.1 and the intrinsic reward oscillating around 0 (learning-progress signal,
     cf. paper Fig 2) instead of a constant −6.
  The pre-fix raw form is preserved verbatim via `LPMModel(reward_space="raw")` (eta·exp − MSE,
  0.5 clip, no gating, lr 1e-2, cadence 5). New tests in `tests/test_models.py`:
  `test_lpm_reward_is_logspace_difference_eq3`, `test_lpm_raw_mode_reproduces_clipped_notebook_form`,
  `test_lpm_error_model_lr_is_paper_consistent` (the old `<=0.5`-bound test was removed — the
  log-space reward is unbounded above). `run_grid.py` gained a **`uniform`** pseudo-method
  (→ `train_maze --method none --random-policy`) so all 6 methods launch from one grid. The earlier
  C.2 50k×**10**-seed RAW-LPM results were moved to `results_rawlpm_10seed/` (+ positions/figures);
  the corrected **64-seed × 6-method × 3-variant** grid (lpm/rnd/icm/mse/none/uniform, 50k steps)
  was run fresh into `results/`,`positions/`. Only LPM's code changed — rnd/icm/mse/none/uniform
  are byte-for-byte the prior methods. NB: pytest had to be added to the venv (`uv pip install pytest`).

- **2026-06-17 — relocated all experiment artifacts + datasets to `/expr_data/` (repo
  root, gitignored).** To keep the package tree clean and centralise every bulky local
  artifact, the experiment data was physically moved (renames, contents unchanged) out
  of `LPM_exploration/` into a single top-level `expr_data/`:
  - `Miniworld/experiments/{results,positions,figures}/` → `expr_data/miniworld/{results,positions,figures}/`
    (current 64-seed × 6-method × 3-variant grid).
  - `Miniworld/experiments/{results,positions,figures}_rawlpm_10seed/` →
    `expr_data/miniworld/rawlpm_10seed/{results,positions,figures}/` (the older 10-seed RAW-LPM snapshot).
  - `Miniworld/data/` (the 341 MB CIFAR-10 download) → `expr_data/datasets/cifar-10/`.
  Code repointed to the new home (these are *our* harness scripts, not upstream):
  `Miniworld/experiments/run_grid.py` (RESULTS/POSITIONS now under `<repo>/expr_data/miniworld`),
  `Miniworld/experiments/analyze.py` (default `--results/--positions/--figures`),
  `Atari/experiments/run_grid.py` (`--results-dir` default) and `Atari/experiments/analyze.py`
  (RES/OUT) now default to `<repo>/expr_data/atari/{results,figures}` (no Atari data exists yet —
  repointed for consistency). The upstream CIFAR loader `Atari/exploration/cifar.py` is left
  untouched: it still uses `root='./data'` (CWD-relative), so a fresh Atari noisy-TV run would
  re-download rather than read `expr_data/datasets/cifar-10/`. `.gitignore` gained `/expr_data/`
  (the old per-dir artifact rules were kept as defensive backstops).
