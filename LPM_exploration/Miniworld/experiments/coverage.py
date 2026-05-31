"""Maze-geometry coverage helpers. Pure numpy; no torch, no env."""
from __future__ import annotations

import numpy as np

NX, NZ = 72, 48           # grid cells (18*4, 12*4)
CELL = 0.25               # world units per cell
# (min_x, max_x, min_z, max_z) per room; room3 is the thin noise wall.
ROOMS = [
    (0.0, 4.0, 0.0, 8.0),     # room1
    (14.0, 18.0, 0.0, 8.0),   # room2
    (0.0, 18.0, 8.0, 8.1),    # room3 (noise wall)
    (0.0, 18.0, 8.1, 12.0),   # room4
]
WALL_Z = 8.0
WALL_BAND = (7.5, 8.5)    # "fascination with noise" band


def to_cell(x: float, z: float) -> tuple[int, int]:
    ix = min(NX - 1, max(0, int(x * 4)))
    iz = min(NZ - 1, max(0, int(z * 4)))
    return ix, iz


def _cell_centers():
    xs = (np.arange(NX) + 0.5) * CELL
    zs = (np.arange(NZ) + 0.5) * CELL
    return xs, zs


def reachable_mask() -> np.ndarray:
    """Cells whose centre lies inside any room rectangle."""
    xs, zs = _cell_centers()
    X, Z = np.meshgrid(xs, zs, indexing="ij")   # (NX, NZ)
    m = np.zeros((NX, NZ), dtype=bool)
    for x0, x1, z0, z1 in ROOMS:
        m |= (X >= x0) & (X < x1) & (Z >= z0) & (Z < z1)
    return m


def beyond_wall_mask() -> np.ndarray:
    """Reachable cells past the noise wall (room4, z >= 8.1)."""
    xs, zs = _cell_centers()
    X, Z = np.meshgrid(xs, zs, indexing="ij")
    return reachable_mask() & (Z >= 8.1)


def reachable_count() -> int:
    return int(reachable_mask().sum())


def occupancy_grid(xs: np.ndarray, zs: np.ndarray) -> np.ndarray:
    """Count of visits per cell from position arrays."""
    g = np.zeros((NX, NZ), dtype=np.int64)
    ix = np.clip((np.asarray(xs) * 4).astype(int), 0, NX - 1)
    iz = np.clip((np.asarray(zs) * 4).astype(int), 0, NZ - 1)
    np.add.at(g, (ix, iz), 1)
    return g


def coverage_metrics(xs: np.ndarray, zs: np.ndarray) -> dict:
    g = occupancy_grid(xs, zs)
    visited = g > 0
    reach = reachable_mask()
    bw = beyond_wall_mask()
    reach_n = int(reach.sum())
    bw_n = int(bw.sum())
    zs = np.asarray(zs)
    in_band = (zs >= WALL_BAND[0]) & (zs <= WALL_BAND[1])
    return {
        "visited_count": int(visited.sum()),
        "coverage_frac": float((visited & reach).sum() / reach_n),
        "beyond_wall_frac": float((visited & bw).sum() / bw_n) if bw_n else 0.0,
        "time_at_wall_frac": float(in_band.mean()) if len(zs) else 0.0,
    }
