"""Occupancy/coverage heatmaps over training-progress windows, with maze overlay."""
from __future__ import annotations

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402
from matplotlib.colors import LogNorm  # noqa: E402

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


def avg_window_occupancy(runs, n_windows=5):
    """Mean per-cell step-count across seeds. runs: list of (steps, xs, zs)."""
    grids = [window_occupancy(s, x, z, n_windows).astype(np.float64) for s, x, z in runs]
    return np.mean(grids, axis=0)


def avg_cumulative_frontier(runs, n_windows=5):
    """Per-cell fraction of seeds that have ever visited the cell by end of each
    window (0 = no seed reached it, 1 = all seeds did)."""
    grids = [cumulative_frontier(s, x, z, n_windows).astype(np.float64) for s, x, z in runs]
    return np.mean(grids, axis=0)


def plot_evolution(per_method, variant, out_path, mode="density", n_windows=5):
    """per_method: dict method -> list of (steps, xs, zs), one entry per seed.
    Grids are averaged across seeds before plotting (density: mean visit-count per
    cell; frontier: fraction of seeds that ever visited the cell). Saves a
    rows(method)×windows grid with a single shared colorbar.

    For density the colour scale is a SHARED log scale across every panel (the raw
    per-cell visit count spans ~0.1 to ~1000, so a per-panel linear scale hides all
    the structure); the colorbar reads as actual visit counts on a log axis."""
    methods = list(per_method)
    n_seeds = max((len(v) for v in per_method.values()), default=1)
    grids = {m: (avg_window_occupancy(per_method[m], n_windows) if mode == "density"
                 else avg_cumulative_frontier(per_method[m], n_windows))
             for m in methods}

    if mode == "density":
        stacked = np.stack(list(grids.values()))
        pos = stacked[stacked > 0]
        vmin = float(pos.min()) if pos.size else 1.0
        vmax = float(stacked.max()) if stacked.size else 1.0
        if vmax <= vmin:
            vmax = vmin * 10.0
        norm = LogNorm(vmin=vmin, vmax=vmax)
        cmap = plt.cm.viridis.copy()
        cmap.set_bad(cmap(0.0)); cmap.set_under(cmap(0.0))  # unvisited cells -> dark
        cbar_label = f"mean visit count per cell over {n_seeds} seeds (log scale)"
        imshow_kw = dict(cmap=cmap, norm=norm)
        kind = "occupancy density (visit count)"
    else:
        cbar_label = f"fraction of {n_seeds} seeds that visited the cell"
        imshow_kw = dict(cmap="viridis", vmin=0.0, vmax=1.0)
        kind = "coverage frontier (visit fraction)"

    fig, axes = plt.subplots(len(methods), n_windows,
                             figsize=(2.2 * n_windows + 0.8, 2.0 * len(methods)),
                             squeeze=False)
    im = None
    for r, m in enumerate(methods):
        for c in range(n_windows):
            ax = axes[r][c]
            data = grids[m][c].T  # (NZ, NX) so z is vertical
            im = ax.imshow(data, origin="lower", extent=[0, 18, 0, 12],
                           aspect="auto", **imshow_kw)
            _overlay(ax)
            if r == 0:
                ax.set_title(f"win {c+1}/{n_windows}", fontsize=8)
            if c == 0:
                ax.set_ylabel(f"{m} (n={len(per_method[m])})", fontsize=9)
            ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(f"{variant} — {kind}, mean over {n_seeds} seeds")
    fig.tight_layout(rect=(0.0, 0.0, 0.9, 0.97))
    cax = fig.add_axes((0.915, 0.12, 0.014, 0.76))
    fig.colorbar(im, cax=cax).set_label(cbar_label, fontsize=9)
    fig.savefig(out_path, dpi=130); plt.close(fig)
