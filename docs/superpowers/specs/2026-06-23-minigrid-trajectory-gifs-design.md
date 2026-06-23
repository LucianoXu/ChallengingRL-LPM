# MiniGrid trajectory GIF gallery — design

Date: 2026-06-23
Status: implemented (with one deviation — see below)
Scope: `minigrid_exp/` (the sparse-reward MiniGrid intrinsic-reward study)

## Implementation note (2026-06-23 deviation)

The plan below assumed **no intermediate checkpoints existed** and therefore
called for re-training a curated subset with explicit snapshotting. That was
inaccurate: the study's chunked driver kept (a) every config's **final** model
(`results/models/ppo/<run>.zip`) and (b) per-chunk **best-eval** checkpoints
(`results/models/ppo/best/<run>/c<step>/best_model.zip`) written by the
`EvalCallback`. What was genuinely missing was only a per-step progression and a
random/untrained baseline.

So the stages are now **sourced from disk, no re-training**:
- **untrained** — a freshly-instantiated (random) policy, saved on demand
  (`gif_gallery._ensure_untrained`); instantiation, not training.
- **mid** — the per-chunk best-eval checkpoint nearest each config's `mid_steps`
  target (c0 excluded, as it is already competent).
- **final** — the study's final model.

`make_stage_snapshots.py` is retained as a tool for clean per-step re-training if
ever wanted, but is not on the default path. Render seeds: the DoorKey trio
shares seed 42; both MultiRoom configs use seed 64 (same layout — `none` fails at
every stage while `rnd` only solves at the final stage, since RND's deterministic
MultiRoom policy solves ~18% of layouts, the documented "unreliable deterministic
policy").

**Noise depiction (revised after review).** The original §3 plan ("true cell
ghosted beneath TV-static") was inaccurate to the observation space and inflated
the apparent noise: the wrapper corrupts each of the 7×7×3 `[object, color,
state]` channel-elements independently at prob 0.10, so a *cell* shows a change
~24% of the time (any of 3 channels) even though only ~9% of cells get a changed
*object*. Rendering static-over-true hid the real failure mode. The egocentric
panel now renders the agent's **actual noisy symbolic observation as MiniGrid
sprites** (`render_ego`, sanitizing out-of-range codes so `Grid.decode` doesn't
raise), so corrupted cells display the **hallucinated object** the agent
perceives (phantom keys, doors, lava, goals). A thin **magenta outline**
(`mark_corruption`) flags the object-hallucinated cells (~9%). This is faithful
to what the policy receives and to the noisy-TV failure mode. Everything else
below is as built.

## Goal

Produce a curated gallery of animated GIFs that visually communicate the two
headline findings of the MiniGrid study, by showing **how a specific trained
agent walks the maze at a specific training stage**. Each GIF must depict:

1. the agent's **trajectory** through the maze (a persistent breadcrumb trail);
2. the agent's **observation range** (its 7×7 field of view); and
3. under noise, the **observation anomalies** the noise wrapper induces — shown
   in the agent's egocentric view, with the true cell ghosted faintly beneath
   the corruption.

The deliverable is a presentation/report asset, not a new experiment. It reuses
the existing trained-policy machinery and the existing env/wrapper stack.

## Non-goals

- Not re-running the full grid or changing any study result.
- Not an interactive viewer (that is `miniworld_play/`, a different setting).
- Not a per-frame reward/intrinsic HUD (explicitly descoped — two panels only).
- Not auto-searching seeds for the "best" episode; one representative seed per
  config, manual bump if the documented outcome doesn't show (see §2).

## Background facts that constrain the design

- The chunked checkpoint→resume driver overwrites a single `<run>.zip` per
  config; **no intermediate-stage snapshots exist on disk**. Per-stage GIFs
  therefore require re-training a curated subset with explicit snapshotting.
- The `ObservationNoiseWrapper` corrupts the *observation array*, not world
  state, so the top-down `env.render()` can **never** show noise. Noise is only
  visible in the agent's egocentric observation panel.
- MiniGrid's `env.render()` already shades the agent's FOV (`highlight=True` by
  default). Full render of DoorKey-5x5 is 160×160 px at `TILE_PIXELS=32`.
- The agent's observation is a (7,7,3) array of `[object_idx, color_idx, state]`.
  Noise replaces channel values with random ints 0–9, producing **out-of-range
  color/state codes**. `Grid.decode()` **raises** on those (verified:
  `KeyError` on color idx 8). Therefore the renderer must decode the *true*
  (always-valid) image and overlay corruption per cell — it never decodes the
  raw noisy array.
- Existing deps suffice: `imageio` (already used by `make_trace.py`), `Pillow`,
  `numpy`. No new dependencies.

## Curated config set (6 configs × 3 stages = 18 GIF series)

Chosen to tell the two documented findings. One representative seed per config.

Finding A — intrinsic motivation is **difficulty-gated** (clean envs):

| # | env | noise | method | story |
|---|-----|-------|--------|-------|
| 1 | MultiRoom-N6 | clean | none | baseline never reaches goal — gate *needed* |
| 2 | MultiRoom-N6 | clean | rnd  | intrinsic solves it — gate *decisive* |
| 3 | DoorKey-5x5  | clean | none | easy env, solves without help — gate *not needed* |

Finding B — observation noise **flips the trade-off** (DoorKey-5x5, the cleanest case):

| # | env | noise | method | story |
|---|-----|-------|--------|-------|
| 4 | DoorKey-5x5 | noisy | none | noisy baseline |
| 5 | DoorKey-5x5 | noisy | rnd  | RND collapses under noise |
| 6 | DoorKey-5x5 | noisy | lpm  | LPM stays robust under noise |

Each config rendered at **3 stages** — untrained (0 steps) → mid → final — all
on the **same fixed maze layout** (a per-config `render_seed`) so the behavioral
arc is directly comparable across stages.

Noise configs use `noise_prob = 0.10` (the study default). Intrinsic configs use
the study's tuned scales (`config.RND_REWARD_SCALE = 0.005`,
`config.LPM_REWARD_SCALE = 0.001`).

### Per-config training budgets

`mid ≈ 1/3 of final`, matching the budgets at which the study observed each
outcome (FINDINGS.md: DoorKey converges fast, MultiRoom-N6 headline at 2M):

| env | final steps | mid steps | expected final outcome |
|-----|-------------|-----------|------------------------|
| DoorKey-5x5 clean none   | 500,000   | 150,000 | solves |
| DoorKey-5x5 noisy none   | 1,000,000 | 300,000 | mostly fails |
| DoorKey-5x5 noisy rnd    | 1,000,000 | 300,000 | collapses (rarely solves) |
| DoorKey-5x5 noisy lpm    | 1,000,000 | 300,000 | solves (robust) |
| MultiRoom-N6 clean none  | 2,000,000 | 700,000 | never reaches goal |
| MultiRoom-N6 clean rnd   | 2,000,000 | 700,000 | solves |

Budgets are config attributes (editable in one place), not hard-coded in the
renderer. All 6 re-trainings run in parallel on the 128-core box.

## Components

Three new modules in `minigrid_exp/`, one new reusable wrapper, one test file.
Each has a single clear purpose and a small interface.

### `wrappers/ego_capture.py` — `EgoCaptureWrapper`

A `gym.ObservationWrapper` over the dict-obs env. `observation()` records
`self.last_image = obs["image"].copy()` and returns the obs **unchanged**.
Placed *after* the noise wrapper and *before* `ImgObsWrapper`, so `last_image`
is exactly the (possibly noisy) 7×7 image the policy received this step. Pure
pass-through — no behavior change to training or eval.

Depends on: `gymnasium`. Testable in isolation (feed a dict obs, assert capture).

### `gif_config.py` — the curated config list

A module-level list `GIF_CONFIGS` of dataclass/dict entries, each with:
`env_id, noise (bool), method, seed, render_seed, final_steps, mid_steps,
expected_outcome (str, for the human check), slug`. Single source of truth for
both the snapshot tool and the renderer. Slugs e.g.
`multiroom-n6_clean_none`, `doorkey-5x5_noisy_lpm`.

### `make_stage_snapshots.py` — train + dump per-stage checkpoints

CLI: `PYTHONPATH=. python make_stage_snapshots.py [--slug ...] [--jobs N]`.

For each config: build the training env/model exactly as `train_agent` does
(reuse its env + algorithm construction so snapshots are faithful to the study
pipeline — refactor the model-building part of `train.py` into a small reusable
helper if needed, without changing its behavior). Then:

1. Build a fresh model, **save `<run>__step0.zip`** (the untrained stage).
2. `model.learn(mid_steps)`, save `<run>__step<mid>.zip`.
3. `model.learn(final_steps − mid_steps)` (continue, no reset), save
   `<run>__step<final>.zip`.

Output dir: `expr_data/minigrid/results/models/ppo_gif_snapshots/`. Run names
follow the existing `<env>__<variant>__<method>__seed_<n>[__np0.1]` scheme with
a `__step<N>` suffix. Idempotent: skip a snapshot whose zip already exists.

Seed handling: train the config's default seed. After the final snapshot, the
tool prints the deterministic eval return on `render_seed`; if it contradicts
`expected_outcome`, the human bumps `seed` in `gif_config.py` and re-runs that
one config (no automated multi-seed search — YAGNI).

### `gif_gallery.py` — two-panel renderer + per-config driver

CLI: `PYTHONPATH=. python gif_gallery.py [--slug ...]` → renders all stages of
the requested configs (default: all 6).

**Render env builder** (local to this module, *not* `make_env`, so we can insert
the capture wrapper and keep a handle to the base env). Replicates the eval
wrapper order the models were trained against:
`base = gym.make(env_id, render_mode="rgb_array")`
→ `ObservationNoiseWrapper` (only if noise)
→ `EgoCaptureWrapper`
→ `ImgObsWrapper` → `FlattenObservation`
→ `MiniGridActionSubsetWrapper` (via the existing `get_action_map(env_id)`).
This matches the trained 147-dim ImgObs space and the reduced action set, so
`model.predict` works unchanged. Keeps references to `base` and the capture
wrapper.

**Rollout (per stage):** load `<run>__step<N>.zip`; `reset(seed=render_seed)`;
step until `terminated/truncated` or the env's `max_episode_steps`. Record agent
(x,y) each step for the trail. Do **not** prefer a solved episode — early stages
should show wandering/failure.

Action selection by stage: the **untrained** stage uses **stochastic sampling**
(`deterministic=False`) — a freshly-initialized policy under argmax tends to
emit a constant action and spin in place, which misrepresents "random
wandering"; sampling produces the intended exploratory walk. The **mid** and
**final** stages use `deterministic=True` to show the learned policy's intent.
This per-stage choice is recorded in the stage label so the viewer knows which
is shown.

**Left panel (top-down):** `base.render()` (FOV already shaded). Overlay a
persistent **trajectory trail**: small dots/polyline at visited cell centers,
pixel = `(cell + 0.5) * TILE_PIXELS`, faded oldest→newest, current cell marked.
Optionally a crisp outline on the highlighted FOV region.

**Right panel (egocentric):** each frame:
1. `true_img = base.unwrapped.gen_obs()["image"]` (always valid; pure, no extra
   randomness).
2. `noisy_img = capture.last_image` (== `true_img` for clean configs).
3. Decode + render `true_img` to sprites via `Grid.decode` →
   `grid.render(TILE_PIXELS, agent_pos=(3,6), agent_dir=3, highlight_mask=vis)`
   (the standard egocentric pose: agent bottom-center facing up).
4. `corruption_mask[i,j] = any(noisy_img[i,j] != true_img[i,j])`.
5. For each corrupted cell: dim its true tile toward gray (blend ≈ 0.35 true)
   and overlay a conspicuous "static" marker showing the scrambled value — the
   true cell stays **faintly visible underneath the corruption** (the chosen
   depiction). Clean configs show the plain partial view.

**Frame composition:** scale panels to equal height, hconcat with a thin
divider and small text labels (config + stage). Write per-stage GIF via
`imageio.mimsave(..., fps≈6)`.

**Contact strip (per config):** hconcat the three stage two-panel frames into
one wide frame, padding shorter stages by freezing their last frame to the max
length; column headers "untrained / mid / final". Write `strip.gif`.

### Output layout

```
expr_data/minigrid/figures/gifs/
  <slug>/untrained.gif
  <slug>/mid.gif
  <slug>/final.gif
  <slug>/strip.gif          # 3-stage contact strip
  README.md                 # index: slug → finding, env, noise, method, seed
```

## Testing

`minigrid_exp/tests/test_gif_gallery.py`:

- **Egocentric corruption handling:** build a true (7,7,3) image and a noisy
  copy with out-of-range codes (e.g. color 8); assert the renderer returns an
  RGB array of the expected shape and does **not** raise (it decodes the *true*
  image and overlays), and that corrupted cells differ from the clean render.
- **Corruption mask:** identical true/noisy → all-false mask, plain render.
- **Trail overlay:** given a list of cells, assert dots land at the expected
  pixel centers and the frame shape is unchanged.
- **Frame composition:** hconcat of two panels → expected width/height; contact
  strip pads to max length.
- **EgoCaptureWrapper:** passes obs through unchanged and stores `last_image`.
- **Headless smoke:** render one stage of `doorkey-5x5_clean_none` against an
  **already-trained final model on disk** (no retraining) to validate the full
  renderer cheaply before any snapshotting runs.

## File-by-file change list

New:
- `minigrid_exp/wrappers/ego_capture.py`
- `minigrid_exp/gif_config.py`
- `minigrid_exp/make_stage_snapshots.py`
- `minigrid_exp/gif_gallery.py`
- `minigrid_exp/tests/test_gif_gallery.py`
- `expr_data/minigrid/figures/gifs/README.md` (written by the driver)

Modified:
- `minigrid_exp/train.py` — *only if* a small model-building helper extraction
  is needed for reuse by `make_stage_snapshots.py`; behavior unchanged.
- `minigrid_exp/README.md` — document the new GIF tooling and how to run it.

Untouched: `make_trace.py` (kept as the simple single-shot tool), all study
results, all other wrappers.

## Risks / open points

- **MultiRoom-N6 2M re-train** is the heaviest step (~minutes–tens of minutes
  with 8 subproc envs); acceptable on the box, parallel with the others.
- **Seed not exhibiting the documented outcome:** mitigated by the post-train
  eval print + manual seed bump (§2).
- **Egocentric pose orientation** (agent_pos/agent_dir for `grid.render`)
  verified to produce a correctly-oriented view in the venv; pin it in a test.
- **Trail readability on a 5×5 grid** at 32 px/tile: dots may be large; tune
  dot radius/alpha during implementation, not a design blocker.
