import numpy as np
import coverage as cov


def test_to_cell_basic():
    assert cov.to_cell(0.0, 0.0) == (0, 0)
    assert cov.to_cell(2.0, 1.0) == (8, 4)         # agent spawn
    assert cov.to_cell(17.99, 11.99) == (71, 47)   # far corner, clamped in-range


def test_reachable_mask_excludes_gap_and_counts_rooms():
    m = cov.reachable_mask()
    assert m.shape == (72, 48)
    # The gap between rooms 1 and 2 (x in [4,14], z in [0,8]) is unreachable.
    assert not m[cov.to_cell(9.0, 4.0)]
    # Inside room1, room2, room4 are reachable.
    assert m[cov.to_cell(2.0, 4.0)]
    assert m[cov.to_cell(16.0, 4.0)]
    assert m[cov.to_cell(9.0, 10.0)]
    # Reachable count is stable (regression guard).
    assert int(m.sum()) == cov.reachable_count()


def test_beyond_wall_mask_is_room4_only():
    bw = cov.beyond_wall_mask()
    assert bw[cov.to_cell(9.0, 10.0)]      # room4
    assert not bw[cov.to_cell(2.0, 4.0)]   # room1 (below wall)
    assert (bw & ~cov.reachable_mask()).sum() == 0  # subset of reachable


def test_coverage_metrics_from_positions():
    xs = np.array([2.0, 2.0, 9.0, 9.0])   # two cells in room1, two in room4
    zs = np.array([4.0, 4.0, 10.0, 10.5])
    cm = cov.coverage_metrics(xs, zs)
    assert cm["visited_count"] == 3        # (8,16) once + (36,40)+(36,42)
    assert 0 < cm["coverage_frac"] <= 1
    assert cm["beyond_wall_frac"] > 0
    assert 0 <= cm["time_at_wall_frac"] <= 1
