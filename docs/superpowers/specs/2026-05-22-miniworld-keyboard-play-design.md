# Miniworld Keyboard Play Tool — Design

**Date:** 2026-05-22
**Status:** Approved (Section 1 with the user; Sections 2–6 delegated to autonomous design under the user's explicit "this isn't complex, run autonomously" instruction).

## Goal

Wrap the three Miniworld scenarios from Hou et al. 2026 (the LPM paper) so a human can explore them by keyboard, **in exactly the same way the trained agent would**. Deliverable is a CLI tool that:

1. Reproduces the paper's hand-designed 4-room `MazeEnv` (not `miniworld`'s built-in Hallway/Maze).
2. Supports all three variants: `nonoise`, `action_noise` (CIFAR-10 injection), `noisy_tv` (random-RGB noise wall + 25 % sticky actions).
3. Renders a pygame window with the agent's 160×120 first-person view as the main pane and a side pane with the top-down view, visit heatmap, and live state info.
4. Records every step (action, position, direction, visit count, sticky-replay flag) to a JSONL file.

## Non-goals (YAGNI)

- No trained-agent demo / playback of policy actions in this round.
- No live visualisation of intrinsic-reward signal (would require running MSE/AMA/LPM dynamics models — out of scope).
- No replay of recorded trajectories. JSONL is the persistence format; replay can be added later if needed.

## Architecture & file layout

New top-level directory `miniworld_play/`. We do **not** put the code inside `LPM_exploration/` because that subtree is treated as a third-party upstream snapshot.

```
ChallengingRL/
└── miniworld_play/
    ├── README.md            — install, run, keybindings, CLI flags
    ├── envs.py              — base MazeEnv + 3 variants; gymnasium-registered
    ├── play.py              — pygame UI + keyboard + JSONL recorder + CLI
    └── recordings/          — output dir for JSONL (gitignored)
```

The existing `LPM_exploration/.venv/` (Python 3.11, torch 2.12) is reused. Added dependencies: `gymnasium`, `miniworld`, `pygame`, `Pillow`, `torchvision` (last one needed only for the `action_noise` variant's CIFAR-10 loader; lazy-imported there).

## Env layer (`envs.py`)

Ported faithfully from the three upstream notebooks at `LPM_exploration/Miniworld/`. The `_gen_world()` geometry is identical across variants (it's the 4-room layout with a thin "noise wall" at z ∈ [8.0, 8.1]) and is hoisted into a base class.

```
class MazeEnv(MiniWorldEnv, utils.EzPickle):
    def _gen_world(self):
        # 4 rooms + 3 connections + agent spawn at [2, 0, 1] facing -π/2
        # Wall textures: cardboard / asphalt / grass / default.
        # The grass-textured strip is the "noise wall".
        ...

    def step(self, action):
        # 4 actions: turn_left=0, turn_right=1, move_forward=2, move_back=3.
        # fwd_step=0.4, turn_step=5°.
        # Sticky actions (NoisyTV only) handled here.
        # Reward = 0 (pure exploration). info["visited_state"] = visit count.
```

Three subclasses:

| Subclass        | Obs transform              | Sticky actions | Extra deps   |
|-----------------|----------------------------|----------------|--------------|
| `NoNoiseEnv`    | none                       | no             | —            |
| `ActionNoiseEnv`| green pixels → random CIFAR-10 sample (per-action) | no | torchvision |
| `NoisyTVEnv`    | green pixels → per-pixel random RGB (per-step)     | yes (25 %)  | —            |

The CIFAR-10 dataset is downloaded to `./LPM_exploration/Miniworld/data/` (matches the upstream notebook's path) so it's reused across runs and gitignored.

Each variant calls `register()` with a stable id: `LPMPlay-MazeNoNoise-v0`, `LPMPlay-MazeActionNoise-v0`, `LPMPlay-MazeNoisyTV-v0`. The tool resolves these via `gym.make()`.

## UI layer (`play.py`)

### Window composition

Total window: **1024 × 768**.

```
+------------------------------------+----------------------+
|                                    |                      |
|   Main pane: first-person view     |  Top-down + visited  |
|   (160×120 upscaled 4× → 640×480)  |  heatmap overlay     |
|                                    |  (384×384)           |
|                                    |                      |
|                                    +----------------------+
|                                    |  Status text:        |
|                                    |  variant, step, pos, |
|                                    |  dir, visited, sticky|
|                                    |  ON/OFF, REC ●       |
+------------------------------------+----------------------+
```

The side pane is hideable with `M` for "strict POMDP mode".

### Keyboard mapping

- **Arrow keys** (and WASD as alternative):
  - `Up / W`: move_forward
  - `Down / S`: move_back
  - `Left / A`: turn_left
  - `Right / D`: turn_right
- `R`: reset episode (closes current recording file, starts a new one with ep++)
- `T`: toggle sticky-action stochasticity (default: per-variant — on for `noisy_tv`, off otherwise)
- `M`: toggle side panel ("minimap")
- `SPACE`: pause (don't step env)
- `Q` / `ESC`: quit (flushes recording first)

Key repeat is enabled with `pygame.key.set_repeat(150, 80)` — initial delay 150 ms, repeat 80 ms — so holding a key fires one env step per repeat. This keeps step counts interpretable while still feeling like a real-time game.

### CLI

```bash
python miniworld_play/play.py \
    --variant {nonoise|action_noise|noisy_tv} \    # required
    --seed 0 \                                      # optional, default 0
    --no-record \                                   # optional, skip JSONL
    --recordings-dir miniworld_play/recordings \    # optional
    --max-steps 50000                                # matches paper
```

## Recording schema (JSONL)

File: `recordings/{variant}_{ISO8601 timestamp}.jsonl`.

First line — session metadata:
```json
{"type":"meta","variant":"noisy_tv","sticky_actions":true,"seed":0,
 "max_steps":50000,"started_at":"2026-05-22T11:30:00Z","obs_size":[120,160,3]}
```

Subsequent lines — one per env step:
```json
{"type":"step","ep":0,"t":1,"action":"move_forward","action_id":2,
 "sticky_replayed":false,"pos":[2.0,0.0,1.4],"dir_deg":-90.0,
 "visited_count":2}
```

Frames themselves are NOT stored (would be ~500 MB / 5 min). Position + action + visit-count are enough to (a) replay action sequences against the same env, (b) compute exploration coverage metrics later, (c) compare human vs agent paths.

Reset (`R`) emits a `{"type":"reset","ep":1,...}` event and continues writing to the same file. New session = new file.

## Error handling

- If `miniworld` import fails: print a clear hint pointing at the install command.
- If pygame can't create a display (e.g. SSH session without forwarding): exit with a message about needing a local display, since the rendering is the whole point.
- If `torchvision` is missing only when `action_noise` is requested: lazy-import inside `ActionNoiseEnv.__init__` and surface a clear error.
- CIFAR-10 download failure: fall back to a deterministic synthetic noise pattern so the env still runs (with a printed warning).

## Testing strategy

This is a hands-on interactive tool, so most validation is manual ("I launched it and walked around"). The minimum scripted checks:

1. **Import smoke test** — `python -c "from miniworld_play.envs import NoisyTVEnv; e = NoisyTVEnv(); e.reset()"` for each variant.
2. **Headless render smoke test** — instantiate each variant, call `env.step(2)` 50 times, assert obs.shape == (120, 160, 3) and visit_count > 0.
3. **JSONL roundtrip** — programmatically drive 10 steps through the recorder, then re-parse the file and verify schema.
4. **Manual UI launch** — open each variant, walk through the noise wall, confirm the noise visually behaves as described in the paper, and confirm sticky actions kick in on `noisy_tv`. Save one screenshot per variant.

## What gets committed

In one or two commits:
- `miniworld_play/envs.py`
- `miniworld_play/play.py`
- `miniworld_play/README.md`
- This design doc.
- `.gitignore` updated to exclude `miniworld_play/recordings/` and `LPM_exploration/Miniworld/data/` (CIFAR-10 cache).
- A note in `LPM_exploration/UPSTREAM.md` about the geometry being ported from the notebooks (we are not modifying upstream files; we are extracting from them).
