# `miniworld_play` — Keyboard-controlled play tool for the LPM Miniworld scenes

A small interactive wrapper around the three Miniworld environments from
Hou, An, Du (ICLR 2026), *Beyond Noisy-TVs*, so a human can explore them with
arrow keys and feel what the agent's POMDP actually looks like.

The three variants are reproduced verbatim from the upstream notebooks at
`LPM_exploration/Miniworld/`:

| variant         | noise behaviour                                              | sticky actions |
|-----------------|--------------------------------------------------------------|----------------|
| `nonoise`       | none                                                         | no             |
| `noisy_tv`      | grass-textured "noise wall" → per-pixel random RGB each step | yes (25%)      |
| `action_noise`  | grass-textured wall → random RGB; press **N** for a CIFAR-10 frame | yes (25%) |

The world is a hand-designed 4-room layout (cardboard, asphalt, grass, default);
the agent always spawns at `[2, 0, 1]` facing south. Observation is 160 × 120 RGB.
Reward is always 0 — this is a pure-exploration setup.

## Install

The repo's existing uv venv at `LPM_exploration/.venv` already has everything you
need. If you're starting fresh:

```bash
cd /path/to/ChallengingRL
uv venv --python 3.11 LPM_exploration/.venv
uv pip install --python LPM_exploration/.venv/bin/python \
    gymnasium miniworld pygame Pillow torchvision torch numpy
```

CIFAR-10 is only needed for `action_noise` when you press **N** ("look at the
noisy TV"). The tool reads pre-extracted PNGs straight off disk — it never
downloads during play, because the University-of-Toronto pickle mirror that
`torchvision` uses is frequently throttled to ~1 kB/s and would freeze the UI.
If the images aren't present, `N` silently falls back to synthetic random-RGB
patches. To enable the real frames, fetch the fast.ai mirror once (~135 MB,
~20 MB/s, extracts to `LPM_exploration/Miniworld/data/cifar10/`, gitignored):

```bash
DATA=LPM_exploration/Miniworld/data
curl -L -o "$DATA/cifar10.tgz" https://s3.amazonaws.com/fast-ai-imageclas/cifar10.tgz
tar -xzf "$DATA/cifar10.tgz" -C "$DATA" && rm "$DATA/cifar10.tgz"
```

## Run

```bash
./LPM_exploration/.venv/bin/python miniworld_play/play.py --variant noisy_tv
```

Other CLI flags:

```
--variant {nonoise,noisy_tv,action_noise}   scene to play (default: noisy_tv)
--seed INT                                  reset seed (default: 0)
--no-record                                 don't write a JSONL trace
--recordings-dir DIR                        override output dir
--headless                                  run 50 random steps + save a
                                            screenshot + exit (CI smoke test)
```

## Keybindings

```
Arrows / WASD ........ move_forward / move_back / turn_left / turn_right
N .................... action 4 (look at noisy TV — only meaningful in
                       the action_noise variant; no-op elsewhere)
R .................... reset episode (starts a new recording episode)
T .................... toggle sticky-action stochasticity
M .................... toggle side panel (entering "strict POMDP mode" hides
                       the top-down map, leaving you with only what the agent
                       actually sees)
SPACE ................ pause
F12 .................. save a screenshot to recordings/
Q / ESC .............. quit (flushes recording first)
```

Keys auto-repeat (initial delay 150 ms, repeat 80 ms), so holding a direction
feels smooth while keeping step counts precise — one repeat = one env step.

## Recording schema (JSONL)

Files land at `miniworld_play/recordings/{variant}_{ISO8601}.jsonl`. The
recordings folder is gitignored.

First line is session metadata:

```json
{"type":"meta","variant":"noisy_tv","sticky_prob":0.25,"seed":0,
 "max_steps":50000,"started_at":"2026-05-22T11:30:00Z","obs_size":[120,160,3]}
```

Each subsequent line is one env step:

```json
{"type":"step","ep":0,"t":3,"action":"move_forward","action_id":2,
 "sticky_replayed":true,"pos":[2.0,2.2],"dir_deg":-90.0,"visited_count":3}
```

Pressing **R** emits a `{"type":"reset","ep":1,...}` event and continues
writing to the same file (so one session-file can contain many episodes).

Frames themselves are *not* recorded — only positions, actions, and visit
counts. Replay-against-the-env is therefore action-deterministic but
visually nondeterministic (the noise wall samples fresh randomness on each
replay).

## Files

```
miniworld_play/
├── README.md            — this file
├── envs.py              — base MazeEnv + 3 variants, gymnasium-registered
├── play.py              — pygame UI + keyboard handler + JSONL recorder + CLI
└── recordings/          — output dir (gitignored)
```

## Smoke test

```bash
for v in nonoise noisy_tv action_noise; do
    ./LPM_exploration/.venv/bin/python miniworld_play/play.py \
        --variant $v --headless
done
```

Each invocation runs 50 random actions in `SDL_VIDEODRIVER=dummy` mode and
saves a PNG screenshot of the final frame to `recordings/`. Used in CI / when
checking the tool isn't broken after dependency updates.
