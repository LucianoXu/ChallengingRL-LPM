"""
pacman_play.py — play Ms Pac-Man yourself with the keyboard (pygame).

OUR tool (not upstream). Same spirit as ../../miniworld_play/play.py but for the
Atari Ms Pac-Man env used in the LPM experiments. Opens a pygame window, maps
arrow keys / WASD to the ALE action set, and lets a human play.

Controls:  Arrows or WASD = move (diagonals supported) · R = restart · Q/Esc = quit

Examples
--------
  # play
  PYTHONPATH=. .venv/bin/python pacman_play.py
  # bigger / slower window
  PYTHONPATH=. .venv/bin/python pacman_play.py --scale 4 --fps 12
  # try the noisy-TV variant a human can trigger (idle actions -> random CIFAR frame)
  PYTHONPATH=. .venv/bin/python pacman_play.py --noisy --randop 2
  # headless self-test (no window, for CI / this sandbox)
  PYTHONPATH=. .venv/bin/python pacman_play.py --headless
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import gymnasium as gym
import ale_py
gym.register_envs(ale_py)

class Frameskip(gym.Wrapper):
    """Repeat the chosen action for `skip` game frames (sum reward, last obs).

    Deliberately avoids stable_baselines3.common.atari_wrappers.MaxAndSkipEnv:
    importing it pulls in cv2, which ships a second SDL2 that clashes with
    pygame's on macOS ('mysterious crashes'). This keeps the human player
    dependency-light (gymnasium + ale_py + pygame only).
    """
    def __init__(self, env, skip):
        super().__init__(env)
        self.skip = skip

    def step(self, action):
        total, obs, info, term, trunc = 0.0, None, {}, False, False
        for _ in range(self.skip):
            obs, r, term, trunc, info = self.env.step(action)
            total += r
            if term or trunc:
                break
        return obs, total, term, trunc, info


def build_env(env_name, frameskip, noisy, randop):
    env = gym.make(env_name, render_mode="rgb_array", max_episode_steps=10_000_000)
    if frameskip > 1:
        env = Frameskip(env, frameskip)
    if noisy:
        # lazy import: CIFAR (+cv2) is only loaded when the noisy variant is requested
        from exploration.noisy_wrapper import NoisyTVEnvWrapperCIFAR
        from exploration.cifar import create_cifar_function_simple
        env = NoisyTVEnvWrapperCIFAR(env, create_cifar_function_simple(), randop)
    return env


def _step(env, a):
    out = env.step(a)
    if len(out) == 5:
        o, r, term, trunc, info = out
        return o, r, bool(term or trunc), info
    o, r, d, info = out
    return o, r, d, info


def _reset(env, seed=None):
    out = env.reset(seed=seed) if seed is not None else env.reset()
    return out if isinstance(out, tuple) else (out, {})


def keys_to_action(pressed, name_to_idx, K):
    up = pressed[K.K_UP] or pressed[K.K_w]
    down = pressed[K.K_DOWN] or pressed[K.K_s]
    left = pressed[K.K_LEFT] or pressed[K.K_a]
    right = pressed[K.K_RIGHT] or pressed[K.K_d]
    name = "NOOP"
    if up and right:   name = "UPRIGHT"
    elif up and left:  name = "UPLEFT"
    elif down and right: name = "DOWNRIGHT"
    elif down and left:  name = "DOWNLEFT"
    elif up:    name = "UP"
    elif down:  name = "DOWN"
    elif left:  name = "LEFT"
    elif right: name = "RIGHT"
    if name in name_to_idx:
        return name_to_idx[name]
    # fall back: drop the diagonal to its vertical (then horizontal) component
    for alt in (name.replace("RIGHT", "").replace("LEFT", ""),
                name.replace("UP", "").replace("DOWN", "")):
        if alt in name_to_idx:
            return name_to_idx[alt]
    return name_to_idx.get("NOOP", 0)


def headless_selftest(args):
    env = build_env(args.env_name, args.frameskip, args.noisy, args.randop)
    meanings = env.unwrapped.get_action_meanings()
    print(f"[selftest] env={args.env_name} actions={env.action_space.n} meanings={meanings}")
    obs, info = _reset(env, args.seed)
    score = 0.0
    for i in range(40):
        a = env.action_space.sample()
        obs, r, done, info = _step(env, a)
        score += r
        if done:
            obs, info = _reset(env)
    rgb = env.render()
    print(f"[selftest] stepped 40x, score={score:.0f}, render shape={None if rgb is None else rgb.shape}, lives={info.get('lives')}")
    print("[selftest] OK")
    env.close()


def play(args):
    import pygame
    env = build_env(args.env_name, args.frameskip, args.noisy, args.randop)
    meanings = env.unwrapped.get_action_meanings()
    name_to_idx = {n: i for i, n in enumerate(meanings)}
    n_act_real = env.action_space.n - (args.randop if args.noisy else 0)

    pygame.init()
    pygame.key.set_repeat()  # we poll held keys ourselves
    font = pygame.font.SysFont("monospace", 14, bold=True)
    big = pygame.font.SysFont("monospace", 28, bold=True)
    clock = pygame.time.Clock()

    obs, info = _reset(env, args.seed)
    rgb = env.render()
    H, W = rgb.shape[0] * args.scale, rgb.shape[1] * args.scale
    screen = pygame.display.set_mode((W, H + 26))
    pygame.display.set_caption("Ms Pac-Man — you play")

    score, steps, best, gameover = 0.0, 0, 0, False
    running = True
    while running:
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                running = False
            elif e.type == pygame.KEYDOWN:
                if e.key in (pygame.K_q, pygame.K_ESCAPE):
                    running = False
                elif e.key == pygame.K_r:
                    obs, info = _reset(env); score = 0.0; steps = 0; gameover = False

        if not gameover:
            a = keys_to_action(pygame.key.get_pressed(), name_to_idx, pygame)
            obs, r, done, info = _step(env, a)
            score += r; steps += 1
            best = max(best, int(score))
            if done:
                gameover = True

        rgb = env.render()
        surf = pygame.surfarray.make_surface(np.transpose(rgb, (1, 0, 2)))
        surf = pygame.transform.scale(surf, (W, H))
        screen.fill((0, 0, 0))
        screen.blit(surf, (0, 0))
        screen.blit(font.render(
            f"score {int(score):5d}   lives {info.get('lives','?')}   best {best}", True, (255, 255, 0)), (6, H + 4))
        screen.blit(font.render("Arrows/WASD  R reset  Q quit", True, (170, 170, 170)),
                    (W - 250, H + 4))
        if gameover:
            t = big.render("GAME OVER — R to restart", True, (255, 80, 80))
            screen.blit(t, (W // 2 - t.get_width() // 2, H // 2 - 14))
        pygame.display.flip()
        clock.tick(args.fps)

    env.close()
    pygame.quit()
    print(f"[play] final score {int(score)} (best {best})")


def main():
    p = argparse.ArgumentParser(description="Play Ms Pac-Man yourself (keyboard)")
    p.add_argument("--env-name", default="MsPacmanNoFrameskip-v4")
    p.add_argument("--scale", type=int, default=3, help="window upscale factor")
    p.add_argument("--fps", type=int, default=15, help="decisions/sec (×frameskip = game fps)")
    p.add_argument("--frameskip", type=int, default=4, help="game frames per key decision")
    p.add_argument("--noisy", action="store_true", help="action-noise (CIFAR) variant")
    p.add_argument("--randop", type=int, default=2, help="number of idle/noise actions if --noisy")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--headless", action="store_true", help="no-window self-test")
    args = p.parse_args()
    if args.headless:
        headless_selftest(args)
    else:
        play(args)


if __name__ == "__main__":
    main()
