"""Render trained MiniGrid policies as two-panel trajectory GIFs.

Per frame:
  * LEFT  - top-down maze (env.render, FOV already shaded) + a persistent
            breadcrumb trail of visited agent cells.
  * RIGHT - the agent's egocentric 7x7 observation. Under noise, corrupted cells
            show the TRUE cell ghosted faintly beneath a "static" overlay
            (Grid.decode raises on out-of-range noise codes, so we decode the
            always-valid true image and overlay corruption per cell).

Three stages per config (untrained / mid / final) on a fixed maze layout, plus a
3-stage contact strip. See gif_config.py for the curated set.

Usage:
  PYTHONPATH=. python gif_gallery.py                 # all configs
  PYTHONPATH=. python gif_gallery.py --slug doorkey-5x5_noisy_lpm
  PYTHONPATH=. python gif_gallery.py --smoke <run_name>   # render an on-disk model
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import imageio.v2 as imageio
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import gymnasium as gym  # noqa: E402
from minigrid.wrappers import ImgObsWrapper  # noqa: E402
from minigrid.core.grid import Grid  # noqa: E402
from minigrid.core.constants import TILE_PIXELS  # noqa: E402
from gymnasium.wrappers import FlattenObservation  # noqa: E402

from algorithms import get_algorithm_class  # noqa: E402
from wrappers.noise_wrapper import ObservationNoiseWrapper  # noqa: E402
from wrappers.ego_capture import EgoCaptureWrapper  # noqa: E402
from wrappers.env_factory import MiniGridActionSubsetWrapper, get_action_map  # noqa: E402
from gif_config import GIF_CONFIGS, get_config, GIFS_DIR, NOISE_PROB  # noqa: E402

PANEL_H = 320          # common panel height for both panels (px)
HEADER_H = 30          # label strip height (px)
DIVIDER = 6            # gap between panels (px)
MAX_STEPS = 160        # cap rollout length to bound GIF size
FPS = 8
EGO_AGENT_POS = (3, 6)  # agent at bottom-center of the 7x7 obs, facing up
EGO_AGENT_DIR = 3
TRAIL_RADIUS = 5

try:
    _FONT = ImageFont.truetype("DejaVuSans.ttf", 14)
except OSError:
    _FONT = ImageFont.load_default()


# --------------------------------------------------------------------------- #
# Pure rendering helpers (no env / no model -> directly unit-testable)
# --------------------------------------------------------------------------- #
def _sanitize_for_decode(obs_img: np.ndarray) -> np.ndarray:
    """Clamp noise-injected codes into valid ranges so Grid.decode won't raise.
    The cell-level noise wrapper already draws in-range codes (object 0..10,
    color 0..5, state 0..2), so this clamp is now a safety no-op kept for
    robustness; a corrupted cell still decodes to *some* object (the
    agent's hallucination)."""
    out = obs_img.astype(np.uint8).copy()
    out[..., 0] = np.clip(out[..., 0], 0, 10)   # object
    out[..., 1] = np.clip(out[..., 1], 0, 5)    # color
    out[..., 2] = np.clip(out[..., 2], 0, 2)    # state
    return out


def render_ego(obs_img: np.ndarray, tile: int = TILE_PIXELS) -> np.ndarray:
    """Render a (possibly noisy) 7x7x3 symbolic observation as MiniGrid sprites
    — i.e. exactly what the agent perceives, hallucinated objects and all."""
    grid, _vis = Grid.decode(_sanitize_for_decode(obs_img))
    return grid.render(tile, agent_pos=EGO_AGENT_POS, agent_dir=EGO_AGENT_DIR,
                       highlight_mask=None)


def object_change_mask(true_img: np.ndarray, noisy_img: np.ndarray) -> np.ndarray:
    """(7,7) bool indexed [x,y]: cells whose OBJECT channel the noise altered —
    the visible hallucinations (a wall becomes a key, an empty cell a goal, ...).
    Color/state-only corruptions render faithfully but aren't flagged."""
    if noisy_img is None:
        return np.zeros(true_img.shape[:2], dtype=bool)
    return true_img[..., 0] != noisy_img[..., 0]


def mark_corruption(ego_rgb: np.ndarray, mask: np.ndarray,
                    tile: int = TILE_PIXELS) -> np.ndarray:
    """Thin magenta outline on each object-hallucinated cell. `mask` is indexed
    [x, y]; the tile region is rgb[y.., x..]."""
    out = ego_rgb.copy()
    c = (255, 0, 255)
    xs, ys = np.where(mask)
    for x, y in zip(xs, ys):
        y0, y1, x0, x1 = y * tile, (y + 1) * tile, x * tile, (x + 1) * tile
        out[y0:y0 + 2, x0:x1] = c
        out[y1 - 2:y1, x0:x1] = c
        out[y0:y1, x0:x0 + 2] = c
        out[y0:y1, x1 - 2:x1] = c
    return out


def ego_panel(true_img: np.ndarray, noisy_img: np.ndarray) -> np.ndarray:
    """The egocentric panel: render what the agent sees (noisy if corrupted) and
    flag the object-hallucinated cells."""
    obs_img = noisy_img if noisy_img is not None else true_img
    return mark_corruption(render_ego(obs_img),
                           object_change_mask(true_img, noisy_img))


def draw_trail(topdown_rgb: np.ndarray, cells, tile: int = TILE_PIXELS) -> np.ndarray:
    """Overlay a fading breadcrumb trail of visited (x, y) agent cells.
    Oldest cells faint, current cell marked in cyan."""
    img = Image.fromarray(topdown_rgb.copy())
    draw = ImageDraw.Draw(img, "RGBA")
    n = len(cells)
    for i, (x, y) in enumerate(cells):
        cx, cy = (x + 0.5) * tile, (y + 0.5) * tile
        if i == n - 1:
            color, r = (0, 220, 220, 255), TRAIL_RADIUS + 2   # current = cyan
        else:
            alpha = int(60 + 150 * (i + 1) / max(n - 1, 1))    # fade oldest->newest
            color, r = (255, 80, 0, alpha), TRAIL_RADIUS       # trail = orange
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=color)
    return np.asarray(img)


def _scale_to_height(rgb: np.ndarray, h: int) -> np.ndarray:
    im = Image.fromarray(rgb)
    w = max(1, round(im.width * h / im.height))
    return np.asarray(im.resize((w, h), Image.NEAREST))


def compose_panels(topdown_rgb: np.ndarray, ego_rgb: np.ndarray,
                   title: str) -> np.ndarray:
    """Header strip + [top-down | divider | egocentric], scaled to PANEL_H."""
    left = _scale_to_height(topdown_rgb, PANEL_H)
    right = _scale_to_height(ego_rgb, PANEL_H)
    body_w = left.shape[1] + DIVIDER + right.shape[1]
    body = np.full((PANEL_H, body_w, 3), 255, dtype=np.uint8)
    body[:, :left.shape[1]] = left
    body[:, left.shape[1] + DIVIDER:] = right

    canvas = Image.new("RGB", (body_w, PANEL_H + HEADER_H), (20, 20, 20))
    canvas.paste(Image.fromarray(body), (0, HEADER_H))
    draw = ImageDraw.Draw(canvas)
    draw.text((6, 7), title, fill=(240, 240, 240), font=_FONT)
    return np.asarray(canvas)


def pad_freeze(frames: list[np.ndarray], n: int) -> list[np.ndarray]:
    """Pad a frame list to length n by repeating the last frame."""
    if not frames:
        return frames
    return frames + [frames[-1]] * max(0, n - len(frames))


def contact_strip(stage_frames: dict[str, list[np.ndarray]]) -> list[np.ndarray]:
    """Horizontally concatenate untrained|mid|final frames into one strip."""
    order = [s for s in ("untrained", "mid", "final") if s in stage_frames]
    n = max(len(stage_frames[s]) for s in order)
    padded = {s: pad_freeze(stage_frames[s], n) for s in order}
    out = []
    for t in range(n):
        cols = [padded[s][t] for s in order]
        h = max(c.shape[0] for c in cols)
        cols = [np.pad(c, ((0, h - c.shape[0]), (0, 0), (0, 0)),
                       constant_values=20) for c in cols]
        sep = np.full((h, DIVIDER, 3), 60, dtype=np.uint8)
        row = cols[0]
        for c in cols[1:]:
            row = np.concatenate([row, sep, c], axis=1)
        out.append(row)
    return out


# --------------------------------------------------------------------------- #
# Env / rollout
# --------------------------------------------------------------------------- #
def build_render_env(env_id: str, noise: bool, noise_prob: float, seed: int):
    """Eval-style env matching the trained obs pipeline, with handles to the
    base MiniGrid env (for true obs / render) and the egocentric capture."""
    base = gym.make(env_id, render_mode="rgb_array")
    base.reset(seed=seed)
    env = ObservationNoiseWrapper(base, noise_prob=noise_prob) if noise else base
    capture = EgoCaptureWrapper(env)
    env = FlattenObservation(ImgObsWrapper(capture))
    amap = get_action_map(env_id)
    if amap is not None:
        env = MiniGridActionSubsetWrapper(env, amap)
    return env, base.unwrapped, capture


def rollout(model, env_id: str, noise: bool, render_seed: int,
            title: str, deterministic: bool, noise_prob: float = NOISE_PROB):
    """Run one episode (capped at MAX_STEPS) and return (frames, ep_reward)."""
    env, mg, capture = build_render_env(env_id, noise, noise_prob, render_seed)
    obs, _ = env.reset(seed=render_seed)

    cells = [tuple(int(v) for v in mg.agent_pos)]
    frames, ep_r, done, t = [], 0.0, False, 0
    while not done and t < MAX_STEPS:
        topdown = draw_trail(env.render(), cells)
        ego = ego_panel(mg.gen_obs()["image"], capture.last_image)
        frames.append(compose_panels(topdown, ego, title))

        action, _ = model.predict(obs, deterministic=deterministic)
        obs, r, term, trunc, _ = env.step(action)
        ep_r += float(r)
        done = term or trunc
        cells.append(tuple(int(v) for v in mg.agent_pos))
        t += 1

    # one trailing frame so the final position is visible
    topdown = draw_trail(env.render(), cells)
    ego = ego_panel(mg.gen_obs()["image"], capture.last_image)
    frames.append(compose_panels(topdown, ego, title))
    env.close()
    return frames, ep_r


# --------------------------------------------------------------------------- #
# Drivers
# --------------------------------------------------------------------------- #
def _ensure_untrained(cfg):
    """Save a freshly-instantiated (random) policy for the untrained stage if it
    isn't already on disk. This is instantiation, not training."""
    path = cfg.untrained_path()
    if path.exists():
        return
    from train import get_algorithm_config
    path.parent.mkdir(parents=True, exist_ok=True)
    env, _mg, _cap = build_render_env(cfg.env_id, cfg.noise, NOISE_PROB, cfg.seed)
    algo = get_algorithm_config()
    model = algo["class"](policy=algo["policy"], env=env, verbose=0, seed=cfg.seed,
                          policy_kwargs=algo["policy_kwargs"], **algo["hyperparams"])
    model.save(str(path))
    env.close()
    print(f"  [made] untrained {path.name} (fresh random policy)")


SEED_CANDIDATES = list(range(42, 62))   # 20 candidate layouts for auto-pick


def _eval_return(model, env_id, noise, seed, cap=250):
    env, _mg, _cap = build_render_env(env_id, noise, NOISE_PROB, seed)
    obs, _ = env.reset(seed=seed)
    done, r, t = False, 0.0, 0
    while not done and t < cap:
        a, _ = model.predict(obs, deterministic=True)
        obs, rr, term, trunc, _ = env.step(a)
        r += float(rr); done = term or trunc; t += 1
    env.close()
    return r


def pick_render_seed(cfg) -> int:
    """Choose the fixed layout for a config's three stages. If render_seed is
    pinned, use it. Otherwise scan candidate layouts with the FINAL model and
    show a *solving* episode if the method ever solves (the median-success
    layout, not the luckiest), else the honest failure on the first layout."""
    if cfg.render_seed is not None:
        return cfg.render_seed
    model = get_algorithm_class().load(str(cfg.final_path()))
    rets = {s: _eval_return(model, cfg.env_id, cfg.noise, s) for s in SEED_CANDIDATES}
    solving = sorted((s for s in SEED_CANDIDATES if rets[s] > 0), key=lambda s: rets[s])
    return solving[len(solving) // 2] if solving else SEED_CANDIDATES[0]


def render_config(cfg):
    """Render all three stages + contact strip for one config. Returns
    (final-stage reward, render_seed)."""
    out_dir = GIFS_DIR / cfg.slug
    out_dir.mkdir(parents=True, exist_ok=True)
    _ensure_untrained(cfg)
    seed = pick_render_seed(cfg)
    stage_frames, final_reward = {}, 0.0

    for stage, path, step, deterministic in cfg.stage_sources():
        if not path.exists():
            print(f"  [skip] {cfg.slug}/{stage}: missing model {path}")
            continue
        mode = "deterministic" if deterministic else "sampled"
        if stage == "untrained":
            step_label = "untrained / random"
        elif stage == "mid":
            step_label = f"~{step:,} steps (best-eval)"
        else:
            step_label = f"{step:,} steps"
        title = f"{cfg.slug}  |  {stage}: {step_label} ({mode})"
        model = get_algorithm_class().load(str(path))
        frames, ep_r = rollout(model, cfg.env_id, cfg.noise, seed,
                               title, deterministic)
        imageio.mimsave(out_dir / f"{stage}.gif", frames, fps=FPS, loop=0)
        stage_frames[stage] = frames
        if stage == "final":
            final_reward = ep_r
        print(f"  [ok] {cfg.slug}/{stage}: {len(frames)} frames, reward={ep_r:.3f}")

    if len(stage_frames) >= 2:
        imageio.mimsave(out_dir / "strip.gif",
                        contact_strip({s: stage_frames[s] for s in stage_frames}),
                        fps=FPS, loop=0)
        print(f"  [ok] {cfg.slug}/strip.gif  (layout seed {seed})")
    return final_reward, seed


def write_readme(results: dict):
    lines = [
        "# MiniGrid trajectory GIF gallery",
        "",
        "Two-panel GIFs (top-down trail + FOV / egocentric noise view) at three "
        "training stages per config. See "
        "`docs/superpowers/specs/2026-06-23-minigrid-trajectory-gifs-design.md`.",
        "",
        "Full matrix: 3 envs x {clean, noisy} x {none, entropy, rnd, icm, lpm}. "
        "Each config dir holds `untrained.gif`, `mid.gif`, `final.gif` and a "
        "3-stage `strip.gif`.",
        "",
        "| config | env | noise | method | train seed | layout seed | final return | finding |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for c in GIF_CONFIGS:
        rs = results.get(c.slug)
        fr_s, sd_s = ("-", "-")
        if rs is not None:
            fr, sd = rs
            fr_s, sd_s = f"{fr:.3f}", str(sd)
        lines.append(f"| `{c.slug}` | {c.env_id} | "
                     f"{'yes' if c.noise else 'no'} | {c.method} | {c.seed} | "
                     f"{sd_s} | {fr_s} | {c.finding} |")
    lines += [
        "",
        "Layout seed is auto-picked per config: a solving episode is shown where "
        "the method ever solves (the median-success layout), else the honest "
        "failure. All three stages share that one layout.",
        "",
        "Stages are sourced from existing on-disk checkpoints (no re-training): "
        "**untrained** = a freshly-instantiated random policy; **mid** = the "
        "per-chunk best-eval checkpoint nearest the config's mid target "
        "(`results/models/ppo/best/<run>/c<step>/best_model.zip`); **final** = "
        "the study's final model (`results/models/ppo/<run>.zip`).",
        "",
        f"Noise prob = {NOISE_PROB:g}. Untrained uses sampled actions (a fresh "
        "argmax policy just spins); mid/final are deterministic.",
        "",
        "**Egocentric panel** shows the agent's actual (possibly noisy) 7×7 "
        "symbolic observation rendered as MiniGrid sprites — under noise, "
        "corrupted cells become *hallucinated objects* (phantom keys, doors, "
        "lava, goals), which is the noisy-TV failure mode itself. The noise "
        f"wrapper corrupts each of the 49 cells independently with prob "
        f"{NOISE_PROB:g} (per-*cell*, not per-element), re-drawing all "
        "three channels of a hit cell within their valid ranges (object 0–10, "
        "color 0–5, state 0–2); a **magenta outline** flags cells whose "
        "*object* channel was altered (so on average ~10% of cells are hit).",
        "",
    ]
    (GIFS_DIR / "README.md").write_text("\n".join(lines))
    print(f"wrote {GIFS_DIR / 'README.md'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slug", nargs="+", default=None,
                    help="configs to render (default: all)")
    ap.add_argument("--smoke", default=None,
                    help="render an on-disk MODELS_DIR run_name (validation only)")
    a = ap.parse_args()

    if a.smoke:
        _smoke(a.smoke)
        return

    GIFS_DIR.mkdir(parents=True, exist_ok=True)
    configs = [get_config(s) for s in a.slug] if a.slug else GIF_CONFIGS
    results = {}
    for cfg in configs:
        print(f"== {cfg.slug} ==")
        try:
            results[cfg.slug] = render_config(cfg)
        except Exception as exc:   # don't let one cell abort the 24-config batch
            print(f"  [FAIL] {cfg.slug}: {type(exc).__name__}: {exc}")
    write_readme(results)


def _smoke(run_name: str):
    """Cheap end-to-end validation against an already-trained study model."""
    from config import MODELS_DIR
    import re
    m = re.match(r"(?P<env>.+?)__(?P<variant>\w+?_no_noise|\w+?_noise)__", run_name)
    env_id = m.group("env")
    noise = m.group("variant").endswith("_noise") and not m.group("variant").endswith("no_noise")
    model = get_algorithm_class().load(str(MODELS_DIR / f"{run_name}.zip"))
    frames, ep_r = rollout(model, env_id, noise, 42, f"SMOKE {run_name}",
                           deterministic=True)
    out = GIFS_DIR / "_smoke.gif"
    out.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(out, frames, fps=FPS, loop=0)
    print(f"smoke ok: {out} ({len(frames)} frames, reward={ep_r:.3f}, noise={noise})")


if __name__ == "__main__":
    main()
