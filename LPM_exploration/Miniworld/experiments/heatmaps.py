"""Occupancy/coverage heatmaps over training-progress windows, with maze overlay."""
from __future__ import annotations

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

import coverage as cov  # noqa: E402


def _bin(xs, zs):
    ix = np.clip((np.asarray(xs) * 4).astype(int), 0, cov.NX - 1)
    iz = np.clip((np.asarray(zs) * 4).astype(int), 0, cov.NZ - 1)
    return ix, iz


def window_occupancy(steps, xs, zs, n_windows=5):
    """(n_windows, NX, NZ) step-count per cell within each step window."""
    steps = np.asarray(steps)
    edges = np.linspace(steps.min(), steps.max() + 1, n_windows + 1)
    occ = np.zeros((n_windows, cov.NX, cov.NZ), dtype=np.int64)
    ix, iz = _bin(xs, zs)
    w = np.clip(np.searchsorted(edges, steps, side="right") - 1, 0, n_windows - 1)
    for k in range(n_windows):
        sel = w == k
        np.add.at(occ[k], (ix[sel], iz[sel]), 1)
    return occ


def cumulative_frontier(steps, xs, zs, n_windows=5):
    """(n_windows, NX, NZ) binary: cells ever visited up to end of each window."""
    occ = window_occupancy(steps, xs, zs, n_windows)
    fro = np.zeros_like(occ)
    seen = np.zeros((cov.NX, cov.NZ), dtype=bool)
    for k in range(n_windows):
        seen = seen | (occ[k] > 0)
        fro[k] = seen.astype(np.int64)
    return fro


def _overlay(ax):
    # Draw the reachable rooms (skip the thin noise-wall room3) + the wall line.
    for x0, x1, z0, z1 in cov.ROOMS[:2] + [cov.ROOMS[3]]:
        ax.add_patch(Rectangle((x0, z0), x1 - x0, z1 - z0, fill=False, ec="white", lw=0.8))
    ax.axhline(cov.WALL_Z, color="red", lw=1.0, ls="--")
    ax.set_xlim(0, 18); ax.set_ylim(0, 12)


def plot_evolution(per_method, variant, out_path, mode="density", n_windows=5):
    """per_method: dict method -> (steps, xs, zs). Saves a rows×windows grid."""
    methods = list(per_method)
    fig, axes = plt.subplots(len(methods), n_windows,
                             figsize=(2.2 * n_windows, 2.0 * len(methods)), squeeze=False)
    for r, m in enumerate(methods):
        steps, xs, zs = per_method[m]
        grid = (window_occupancy(steps, xs, zs, n_windows) if mode == "density"
                else cumulative_frontier(steps, xs, zs, n_windows))
        for c in range(n_windows):
            ax = axes[r][c]
            data = grid[c].T  # (NZ, NX) so z is vertical
            disp = np.log1p(data) if mode == "density" else data
            ax.imshow(disp, origin="lower", extent=[0, 18, 0, 12],
                      aspect="auto", cmap="viridis")
            _overlay(ax)
            if r == 0:
                ax.set_title(f"win {c+1}/{n_windows}", fontsize=8)
            if c == 0:
                ax.set_ylabel(m, fontsize=9)
            ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(f"{variant} — {'occupancy density' if mode == 'density' else 'coverage frontier'}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130); plt.close(fig)
