import os
import csv

import numpy as np
import analyze


def _fake_run(results, positions, rid):
    with open(os.path.join(results, rid + ".csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["update", "frames", "coverage_frac", "beyond_wall_frac", "time_at_wall_frac"])
        for u in range(1, 11):
            w.writerow([u, u * 64, 0.01 * u, 0.005 * u, 0.3])
    n = 640
    np.savez_compressed(os.path.join(positions, rid + ".npz"),
                        step=np.arange(n, dtype=np.int32),
                        x=np.random.uniform(0, 18, n).astype(np.float32),
                        z=np.random.uniform(0, 12, n).astype(np.float32))


def test_analyze_produces_table_and_figures(tmp_path):
    results = tmp_path / "results"; positions = tmp_path / "positions"
    figures = tmp_path / "figures"
    results.mkdir(); positions.mkdir()
    for m in ["lpm", "rnd"]:
        for v in ["nonoise", "noisy_tv"]:
            _fake_run(str(results), str(positions), f"{m}-{v}-s1")
    analyze.run(str(results), str(positions), str(figures))
    assert (figures / "table_coverage.csv").exists()
    assert (figures / "fig_coverage_curves.png").exists()
    assert (figures / "fig_beyond_wall.png").exists()
    assert (figures / "fig_heatmap_evolution_noisy_tv_density.png").exists()
