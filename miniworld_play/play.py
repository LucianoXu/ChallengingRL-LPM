"""Keyboard-controlled play tool for the LPM Miniworld scenarios.

Run examples:

    # the headline noisy-TV scenario (25% sticky actions, random-RGB noise wall)
    python miniworld_play/play.py --variant noisy_tv

    # baseline without noise or sticky actions
    python miniworld_play/play.py --variant nonoise

    # action-conditioned noise (press N to "look at the noisy TV")
    python miniworld_play/play.py --variant action_noise

    # headless smoke test: 50 random steps, save a screenshot, exit
    python miniworld_play/play.py --variant noisy_tv --headless

Keybindings:

    Arrows / WASD ........ move_forward / move_back / turn_left / turn_right
    N .................... action 4 (look at noisy TV — only meaningful in
                           the action_noise variant; no-op elsewhere)
    R .................... reset episode (starts a new recording episode)
    T .................... toggle sticky-action stochasticity
    M .................... toggle side panel ("strict POMDP" mode)
    SPACE ................ pause
    F12 .................. save screenshot to recordings/
    Q / ESC .............. quit
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import random
import sys
from pathlib import Path
from typing import Optional

# SDL_VIDEODRIVER must be set BEFORE pygame is imported, so peek at argv early.
if "--headless" in sys.argv:
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import numpy as np  # noqa: E402

# Make package import work whether the user runs as a script or as a module.
_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR.parent) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR.parent))

import pygame  # noqa: E402
import gymnasium as gym  # noqa: E402

from miniworld_play.envs import VARIANT_TO_ID  # noqa: E402


# ---- Layout constants ------------------------------------------------------
WIN_W, WIN_H = 1024, 768
FP_W, FP_H = 640, 480
FP_X, FP_Y = 0, 0
SIDE_X = FP_W
SIDE_W = WIN_W - FP_W           # 384
MAP_SIZE = 384
STATUS_X, STATUS_Y = SIDE_X + 8, MAP_SIZE + 8

FPS_CAP = 30

# ---- Keyboard → action -----------------------------------------------------
# miniworld actions: 0=turn_left, 1=turn_right, 2=move_forward, 3=move_back
KEY_TO_ACTION = {
    pygame.K_UP: 2,
    pygame.K_w: 2,
    pygame.K_DOWN: 3,
    pygame.K_s: 3,
    pygame.K_LEFT: 0,
    pygame.K_a: 0,
    pygame.K_RIGHT: 1,
    pygame.K_d: 1,
    pygame.K_n: 4,
}
ACTION_NAMES = {0: "turn_left", 1: "turn_right", 2: "move_forward", 3: "move_back", 4: "look_at_tv"}


# ---- Recorder --------------------------------------------------------------
class JSONLRecorder:
    def __init__(self, path: Path, meta: dict) -> None:
        self.path = path
        self.ep = 0
        self._t = 0
        path.parent.mkdir(parents=True, exist_ok=True)
        self._f = path.open("w", buffering=1)
        self._write({"type": "meta", **meta})

    def _write(self, obj: dict) -> None:
        self._f.write(json.dumps(obj) + "\n")

    def log_step(self, action_id: int, info: dict) -> None:
        self._t += 1
        self._write({
            "type": "step",
            "ep": self.ep,
            "t": self._t,
            "action": ACTION_NAMES.get(action_id, str(action_id)),
            "action_id": action_id,
            "sticky_replayed": bool(info.get("sticky_replayed", False)),
            "pos": list(info.get("pos", [])),
            "dir_deg": float(np.degrees(info.get("dir", 0.0))),
            "visited_count": int(info.get("visited_state", 0)),
        })

    def reset_episode(self, info: dict) -> None:
        self.ep += 1
        self._t = 0
        self._write({"type": "reset", "ep": self.ep, "visited_count": int(info.get("visited_state", 0))})

    def close(self) -> None:
        try:
            self._f.close()
        except Exception:
            pass


# ---- Rendering helpers -----------------------------------------------------
def numpy_to_surface(arr: np.ndarray) -> pygame.Surface:
    """HxWx3 uint8 → pygame.Surface."""
    # pygame's surfarray expects WxHx3
    return pygame.surfarray.make_surface(arr.swapaxes(0, 1))


def render_top_view_safely(env: gym.Env) -> Optional[np.ndarray]:
    """env.render_top_view() can crash if called before OpenGL is fully
    ready; fall back to None on error so the main loop keeps working."""
    try:
        unwrapped = env.unwrapped
        img = unwrapped.render_top_view()
        if img is None:
            return None
        # miniworld returns HxWx3 uint8 already
        return np.asarray(img)
    except Exception as e:  # pragma: no cover
        print(f"[warn] render_top_view failed: {e}", file=sys.stderr)
        return None


def render_frame(
    screen: pygame.Surface,
    font: pygame.font.Font,
    big_font: pygame.font.Font,
    obs: np.ndarray,
    info: dict,
    state: dict,
    env: gym.Env,
    show_side: bool,
) -> None:
    screen.fill((20, 20, 24))

    # Main pane: first-person obs, scaled.
    fp_surf = numpy_to_surface(obs)
    fp_surf = pygame.transform.scale(fp_surf, (FP_W, FP_H))
    screen.blit(fp_surf, (FP_X, FP_Y))

    # Status badges over the FP view
    badge_y = 8
    badge_x = 8
    if state["paused"]:
        screen.blit(big_font.render("PAUSED", True, (255, 200, 50)), (badge_x, badge_y))
        badge_y += 24
    if state["recording"]:
        pygame.draw.circle(screen, (220, 60, 60), (badge_x + 7, badge_y + 10), 6)
        screen.blit(font.render("REC", True, (220, 200, 200)), (badge_x + 18, badge_y + 4))

    # Side panel
    if show_side:
        pygame.draw.rect(screen, (35, 35, 40), (SIDE_X, 0, SIDE_W, WIN_H))

        top = render_top_view_safely(env)
        if top is not None:
            top_surf = numpy_to_surface(top)
            top_surf = pygame.transform.scale(top_surf, (MAP_SIZE, MAP_SIZE))
            screen.blit(top_surf, (SIDE_X, 0))
        else:
            screen.blit(font.render("(top-down view unavailable)", True, (200, 200, 200)),
                        (SIDE_X + 12, MAP_SIZE // 2))

        # Status text below the map
        lines = [
            f"Variant       : {state['variant']}",
            f"Step          : {state['step_count']}",
            f"Episode       : {state['episode']}",
            f"Pos (x,z)     : ({info.get('pos', [0,0])[0]:.2f}, {info.get('pos', [0,0])[1]:.2f})",
            f"Dir           : {np.degrees(info.get('dir', 0.0)):.1f}°",
            f"Visited cells : {info.get('visited_state', 0)}",
            f"Sticky actions: {'ON' if state['sticky_on'] else 'OFF'}",
            f"Last action   : {state['last_action_str']}",
            f"Sticky kicked : {'YES' if state['last_sticky'] else 'no'}",
            "",
            "Keys: Arrows/WASD move    N=look-at-TV",
            "      R=reset    T=sticky    M=minimap",
            "      SPACE=pause   F12=screenshot   Q=quit",
        ]
        for i, line in enumerate(lines):
            color = (230, 230, 230) if not line.startswith("Sticky kicked") or not state["last_sticky"] \
                else (255, 180, 60)
            screen.blit(font.render(line, True, color), (STATUS_X, STATUS_Y + i * 18))


# ---- Main loop -------------------------------------------------------------
def build_env(variant: str, seed: int) -> gym.Env:
    env = gym.make(VARIANT_TO_ID[variant], render_mode=None)
    env.reset(seed=seed)
    return env


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--variant", choices=list(VARIANT_TO_ID.keys()), default="noisy_tv")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-record", action="store_true")
    p.add_argument("--recordings-dir", default=str(_THIS_DIR / "recordings"))
    p.add_argument("--headless", action="store_true",
                   help="run 50 random steps, save a screenshot to recordings/, exit")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    env = build_env(args.variant, args.seed)
    obs, info = env.reset(seed=args.seed)

    pygame.init()
    pygame.display.set_caption(f"LPM Miniworld Play — {args.variant}")
    pygame.key.set_repeat(150, 80)
    screen = pygame.display.set_mode((WIN_W, WIN_H))
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("Menlo,Monaco,Courier", 14)
    big_font = pygame.font.SysFont("Menlo,Monaco,Courier", 18, bold=True)

    recordings_dir = Path(args.recordings_dir)
    recorder: Optional[JSONLRecorder] = None
    if not args.no_record:
        timestamp = dt.datetime.now().strftime("%Y%m%dT%H%M%S")
        recorder = JSONLRecorder(
            recordings_dir / f"{args.variant}_{timestamp}.jsonl",
            meta={
                "variant": args.variant,
                "seed": args.seed,
                "max_steps": 50000,
                "started_at": dt.datetime.now(dt.timezone.utc).isoformat(),
                "obs_size": list(obs.shape),
                "sticky_prob": getattr(env.unwrapped, "sticky_prob", 0.0),
            },
        )

    state = {
        "variant": args.variant,
        "paused": False,
        "recording": recorder is not None,
        "sticky_on": getattr(env.unwrapped, "sticky_prob", 0.0) > 0.0,
        "step_count": 0,
        "episode": 0,
        "last_action_str": "—",
        "last_sticky": False,
    }
    show_side = True

    running = True
    headless_budget = 50 if args.headless else None
    screenshot_path = recordings_dir / f"{args.variant}_screenshot_{dt.datetime.now().strftime('%Y%m%dT%H%M%S')}.png"

    try:
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key in (pygame.K_q, pygame.K_ESCAPE):
                        running = False
                    elif event.key == pygame.K_r:
                        obs, info = env.reset()
                        if recorder is not None:
                            recorder.reset_episode(info)
                        state["episode"] += 1
                        state["step_count"] = 0
                        state["last_action_str"] = "(reset)"
                        state["last_sticky"] = False
                    elif event.key == pygame.K_t:
                        if hasattr(env.unwrapped, "sticky_prob"):
                            env.unwrapped.sticky_prob = 0.0 if env.unwrapped.sticky_prob > 0 else 0.25
                            state["sticky_on"] = env.unwrapped.sticky_prob > 0
                    elif event.key == pygame.K_m:
                        show_side = not show_side
                    elif event.key == pygame.K_SPACE:
                        state["paused"] = not state["paused"]
                    elif event.key == pygame.K_F12:
                        recordings_dir.mkdir(parents=True, exist_ok=True)
                        pygame.image.save(screen, str(screenshot_path))
                        print(f"[info] screenshot → {screenshot_path}")
                    elif event.key in KEY_TO_ACTION and not state["paused"]:
                        action = KEY_TO_ACTION[event.key]
                        obs, r, term, trunc, info = env.step(action)
                        if recorder is not None:
                            recorder.log_step(action, info)
                        state["step_count"] += 1
                        state["last_action_str"] = ACTION_NAMES[action]
                        state["last_sticky"] = bool(info.get("sticky_replayed", False))
                        if term or trunc:
                            obs, info = env.reset()
                            if recorder is not None:
                                recorder.reset_episode(info)
                            state["episode"] += 1
                            state["step_count"] = 0

            # Headless smoke test: take a few random actions, then snapshot + quit.
            if headless_budget is not None and headless_budget > 0:
                action = random.choice([0, 1, 2, 3])
                obs, r, term, trunc, info = env.step(action)
                if recorder is not None:
                    recorder.log_step(action, info)
                state["step_count"] += 1
                state["last_action_str"] = ACTION_NAMES[action]
                state["last_sticky"] = bool(info.get("sticky_replayed", False))
                headless_budget -= 1
                if headless_budget == 0:
                    render_frame(screen, font, big_font, obs, info, state, env, show_side)
                    pygame.display.flip()
                    recordings_dir.mkdir(parents=True, exist_ok=True)
                    pygame.image.save(screen, str(screenshot_path))
                    print(f"[info] headless: 50 random steps done, screenshot → {screenshot_path}")
                    running = False
                    continue

            render_frame(screen, font, big_font, obs, info, state, env, show_side)
            pygame.display.flip()
            clock.tick(FPS_CAP)
    finally:
        if recorder is not None:
            recorder.close()
        env.close()
        pygame.quit()

    return 0


if __name__ == "__main__":
    sys.exit(main())
