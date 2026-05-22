#!/usr/bin/env python3
"""Render slide assets: env screenshots + LaTeX formula PNGs.

Outputs into ./assets/.  Regenerate after editing with:
    python3 render_assets.py
"""
from pathlib import Path
import os
import sys

ASSETS = Path(__file__).parent / "assets"
ASSETS.mkdir(exist_ok=True)

# ── Formulas (matplotlib mathtext, no LaTeX install needed) ─────────────
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

matplotlib.rcParams["mathtext.fontset"] = "cm"  # Computer Modern, paper-clean


def render_formula(latex: str, path: Path, figsize=(8, 1.2), fontsize=28, color="#1A1A1A"):
    fig, ax = plt.subplots(figsize=figsize, dpi=220)
    ax.text(0.5, 0.5, latex, fontsize=fontsize, ha="center", va="center", color=color)
    ax.axis("off")
    fig.savefig(path, bbox_inches="tight", pad_inches=0.1, transparent=True)
    plt.close(fig)
    print(f"  formula → {path.name}")


print("Rendering formulas …")
render_formula(
    r"$r_{\mathrm{int}}(s,a)\;=\;\| f_\theta(s,a) - s'\,\|^{2}$",
    ASSETS / "formula_curiosity.png",
)
render_formula(
    r"$r_{\mathrm{int}}(s,a)\;\propto\;\mathcal{E}_{\mathrm{prev}}(s,a)\,-\,\mathcal{E}_{\mathrm{curr}}(s,a)$",
    ASSETS / "formula_lpm.png",
    color="#1E4E8C",
)
render_formula(
    r"$r_{\mathrm{int}}^{\mathrm{LPM}}\;\geq\;\frac{1}{c}\,\mathrm{IG}(\theta;\,D)$",
    ASSETS / "formula_ig_bound.png",
    figsize=(7, 1.1),
    fontsize=24,
)


# ── MiniGrid screenshot ─────────────────────────────────────────────────
print("Rendering MiniGrid …")
try:
    import gymnasium as gym
    import minigrid  # noqa: F401  (registers envs)
    from PIL import Image
    import numpy as np

    env = gym.make("MiniGrid-DoorKey-8x8-v0", render_mode="rgb_array")
    env.reset(seed=42)
    # Step a few times to make scene more visually interesting (agent off start cell)
    for _ in range(3):
        env.step(env.action_space.sample())
    frame = env.render()
    Image.fromarray(frame).save(ASSETS / "minigrid_doorkey.png")
    env.close()
    print(f"  → minigrid_doorkey.png  ({frame.shape})")
except Exception as e:
    print(f"  ✗ MiniGrid render failed: {e}", file=sys.stderr)


# ── MiniWorld screenshot ────────────────────────────────────────────────
print("Rendering MiniWorld …")
# MiniWorld needs an OpenGL context.  On macOS the EGL off-screen path is not
# available; pyglet's default (CGL via shadow window) works for windowed-style
# rendering even when called from a script — keep PYGLET_HEADLESS unset.
os.environ.pop("PYGLET_HEADLESS", None)

try:
    import gymnasium as gym
    import miniworld  # noqa: F401
    from PIL import Image

    env = gym.make("MiniWorld-MazeS3-v0", render_mode="rgb_array")
    env.reset(seed=42)
    for _ in range(2):
        env.step(env.action_space.sample())
    frame = env.render()  # (H, W, 3) uint8
    Image.fromarray(frame).save(ASSETS / "miniworld_mazes3.png")
    env.close()
    print(f"  → miniworld_mazes3.png  ({frame.shape})")
except Exception as e:
    print(f"  ✗ MiniWorld render failed: {e}", file=sys.stderr)
    print("    (Slide will gracefully skip image; check Pyglet/OpenGL setup.)",
          file=sys.stderr)


# ── List products ───────────────────────────────────────────────────────
print("\nAssets in", ASSETS.resolve())
for p in sorted(ASSETS.iterdir()):
    print(f"  {p.name}  ({p.stat().st_size // 1024} KB)")
