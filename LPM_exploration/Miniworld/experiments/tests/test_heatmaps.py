import numpy as np
import heatmaps


def test_window_occupancy_shapes():
    steps = np.arange(1000)
    xs = np.random.uniform(0, 18, 1000)
    zs = np.random.uniform(0, 12, 1000)
    occ = heatmaps.window_occupancy(steps, xs, zs, n_windows=5)
    assert occ.shape == (5, 72, 48)
    assert occ.sum() == 1000


def test_cumulative_frontier_is_monotone():
    steps = np.arange(1000)
    xs = np.random.uniform(0, 18, 1000)
    zs = np.random.uniform(0, 12, 1000)
    fro = heatmaps.cumulative_frontier(steps, xs, zs, n_windows=5)
    counts = [(fro[i] > 0).sum() for i in range(5)]
    assert counts == sorted(counts)  # coverage only grows
