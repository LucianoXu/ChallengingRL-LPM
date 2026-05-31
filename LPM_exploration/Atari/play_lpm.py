"""
play_lpm.py — watch a trained LPM (ppo-improvement) agent play Ms Pac-Man.

OUR tool (not upstream). Loads a saved policy checkpoint, runs episodes with the
EXACT training preprocessing (reuses exploration.envs.ProcessFrame84 + the SB3
NoopReset/MaxAndSkip wrappers, 4-frame stack, and the checkpoint's own obs
mean/std normalization), and either:
  * records an animated GIF / MP4 of the gameplay (default), or
  * shows a live pygame window (--live).

Examples
--------
  # record a GIF of 2 episodes
  PYTHONPATH=. PYTORCH_ENABLE_MPS_FALLBACK=1 .venv/bin/python play_lpm.py \
      --model trained_models/lpm_mspacman_1M/MsPacmanNoFrameskip-v4.pt \
      --episodes 2 --out /tmp/lpm_pacman.gif

  # watch live in a window
  ... play_lpm.py --model <ckpt> --live

  # noisy-TV variant (idle actions inject random CIFAR frames)
  ... play_lpm.py --model <ckpt> --noisy --randop 2 --out /tmp/lpm_pacman_noisy.gif
"""
import argparse
import os
import sys

import numpy as np
import torch

# make `exploration` importable when run from the Atari/ dir
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import gymnasium as gym
import ale_py
gym.register_envs(ale_py)

from stable_baselines3.common.atari_wrappers import NoopResetEnv, MaxAndSkipEnv
from exploration.envs import ProcessFrame84
from exploration.model import Policy

try:
    from exploration.noisy_wrapper import NoisyTVEnvWrapperCIFAR
    from exploration.cifar import create_cifar_function_simple
except Exception:
    NoisyTVEnvWrapperCIFAR = None


def build_env(env_name, noisy=False, randop=2, seed=0):
    """Single env with training-matched preprocessing + render_mode='rgb_array'."""
    env = gym.make(env_name, render_mode="rgb_array", max_episode_steps=10_000_000)
    env = NoopResetEnv(env, noop_max=30)
    env = MaxAndSkipEnv(env, skip=4)
    if noisy:
        if NoisyTVEnvWrapperCIFAR is None:
            raise RuntimeError("noisy wrapper unavailable")
        env = NoisyTVEnvWrapperCIFAR(env, create_cifar_function_simple(), randop)
    env = ProcessFrame84(env, crop=False)  # -> (1, 84, 84) channel-first (our fix)
    return env


def load_policy(model_path, env, device):
    ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
    n_act = env.action_space.n
    obs_space = gym.spaces.Box(low=0, high=255, shape=(4, 84, 84), dtype=np.float32)
    act_space = gym.spaces.Discrete(n_act)
    policy = Policy(obs_space, act_space, base_kwargs={"recurrent": False})
    policy.load_state_dict(ckpt["actor_critic_state_dict"])
    policy.to(device).eval()

    # obs normalization used during training
    mean, std = 0.0, 1.0
    rms = ckpt.get("obs_rms", None)
    if rms is not None:
        mean = np.asarray(getattr(rms, "mean")).astype(np.float32)
        std = float(np.sqrt(np.asarray(getattr(rms, "var")).mean()))
    elif ckpt.get("obs_mean", None) is not None:
        mean = np.asarray(ckpt["obs_mean"]).astype(np.float32)
        std = float(np.asarray(ckpt["obs_std"]).mean())
    mean_t = torch.as_tensor(mean, dtype=torch.float32, device=device)
    return policy, mean_t, std, n_act


def overlay(rgb, scale, lines):
    """Upscale RGB (nearest) and draw info lines, returns uint8 HxWx3."""
    from PIL import Image, ImageDraw
    im = Image.fromarray(rgb).resize((rgb.shape[1] * scale, rgb.shape[0] * scale), Image.NEAREST)
    d = ImageDraw.Draw(im)
    y = 2
    for ln in lines:
        d.text((4, y), ln, fill=(255, 255, 0))
        d.text((5, y), ln, fill=(255, 255, 0))  # faux-bold
        y += 12
    return np.asarray(im)


def step_obs(env, action):
    out = env.step(action)
    if len(out) == 5:
        obs, rew, term, trunc, info = out
        done = bool(term or trunc)
    else:
        obs, rew, done, info = out
    return obs, rew, done, info


def reset_obs(env, seed=None):
    out = env.reset(seed=seed) if seed is not None else env.reset()
    return out[0] if isinstance(out, tuple) else out


def run(args):
    device = torch.device("mps") if (not args.cpu and torch.backends.mps.is_available()) else torch.device("cpu")
    env = build_env(args.env_name, noisy=args.noisy, randop=args.randop, seed=args.seed)
    policy, mean_t, std, n_act = load_policy(args.model, env, device)
    print(f"[play] model={os.path.basename(args.model)} actions={n_act} device={device} "
          f"noisy={args.noisy} deterministic={args.deterministic}")

    if args.live:
        import pygame
        pygame.init()
        screen = None
        clock = pygame.time.Clock()

    rnn = torch.zeros(1, policy.recurrent_hidden_state_size, device=device)
    masks = torch.ones(1, 1, device=device)

    frames, ep_scores = [], []
    for ep in range(args.episodes):
        obs = reset_obs(env, seed=args.seed + ep)
        stack = [obs.astype(np.float32)] * 4  # each (1,84,84)
        score, steps = 0.0, 0
        while True:
            x = np.concatenate(stack, axis=0)[None]  # (1,4,84,84)
            xt = (torch.from_numpy(x).to(device) - mean_t) / (std + 1e-8)
            with torch.no_grad():
                _, action, _, rnn = policy.act(xt, rnn, masks, deterministic=args.deterministic)
            a = int(action.item())
            obs, rew, done, _ = step_obs(env, a)
            score += float(rew)
            steps += 1
            stack = stack[1:] + [obs.astype(np.float32)]

            rgb = env.render()
            if rgb is not None:
                tag = "idle/noise" if (args.noisy and a >= n_act - args.randop) else f"act {a}"
                fr = overlay(rgb, args.scale, [f"LPM Ms PacMan  ep {ep+1}/{args.episodes}",
                                              f"score {int(score)}  step {steps}  {tag}"])
                if args.live:
                    import pygame
                    surf = pygame.surfarray.make_surface(np.transpose(fr, (1, 0, 2)))
                    if screen is None:
                        screen = pygame.display.set_mode((fr.shape[1], fr.shape[0]))
                        pygame.display.set_caption("LPM plays Ms Pac-Man")
                    screen.blit(surf, (0, 0)); pygame.display.flip(); clock.tick(args.fps)
                    for e in pygame.event.get():
                        if e.type == pygame.QUIT:
                            done, ep = True, args.episodes
                else:
                    if steps % args.frame_skip == 0:
                        frames.append(fr)

            if done or steps >= args.max_steps:
                break
        ep_scores.append(score)
        print(f"[play] episode {ep+1}: score={int(score)} steps={steps}")

    env.close()
    print(f"[play] mean score over {len(ep_scores)} ep: {np.mean(ep_scores):.1f}")

    if not args.live and frames:
        save_clip(frames, args.out, args.fps)
        print(f"[play] saved {len(frames)} frames -> {args.out}")


def save_clip(frames, out, fps):
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    if out.lower().endswith(".gif"):
        from PIL import Image
        imgs = [Image.fromarray(f) for f in frames]
        imgs[0].save(out, save_all=True, append_images=imgs[1:],
                     duration=int(1000 / fps), loop=0, optimize=True)
    else:
        import cv2
        h, w = frames[0].shape[:2]
        vw = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        for f in frames:
            vw.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
        vw.release()


def main():
    p = argparse.ArgumentParser(description="Watch a trained LPM agent play Ms Pac-Man")
    p.add_argument("--model", required=True, help="path to <env>.pt policy checkpoint")
    p.add_argument("--env-name", default="MsPacmanNoFrameskip-v4")
    p.add_argument("--episodes", type=int, default=2)
    p.add_argument("--out", default="/tmp/lpm_pacman.gif", help=".gif or .mp4 (ignored with --live)")
    p.add_argument("--live", action="store_true", help="show a live pygame window instead of saving")
    p.add_argument("--deterministic", action="store_true", help="argmax actions (default: sample)")
    p.add_argument("--noisy", action="store_true", help="action-noise (CIFAR) variant")
    p.add_argument("--randop", type=int, default=2, help="number of idle/noise actions if --noisy")
    p.add_argument("--scale", type=int, default=3, help="upscale factor for display")
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--frame-skip", type=int, default=1, help="save every Nth frame to GIF/MP4")
    p.add_argument("--max-steps", type=int, default=1200)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--cpu", action="store_true")
    run(p.parse_args())


if __name__ == "__main__":
    main()
