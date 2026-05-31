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
