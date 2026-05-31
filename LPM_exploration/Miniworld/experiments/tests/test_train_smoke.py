import os
import subprocess
import csv

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.dirname(HERE)
PY = os.path.join(EXP, "..", "..", ".venv", "bin", "python")


def test_train_smoke_none_method(tmp_path):
    csv_path = tmp_path / "smoke.csv"
    pos_path = tmp_path / "smoke.npz"
    env = dict(os.environ, PYTHONPATH=EXP, PYTORCH_ENABLE_MPS_FALLBACK="1")
    cmd = [PY, os.path.join(EXP, "train_maze.py"), "--method", "none",
           "--variant", "noisy_tv", "--seed", "0", "--steps", "128",
           "--update-frequency", "64", "--device", "cpu",
           "--csv-log", str(csv_path), "--pos-log", str(pos_path),
           "--log-interval", "1"]
    r = subprocess.run(cmd, env=env, cwd=EXP, capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stderr[-2000:]
    with open(csv_path) as f:
        rows = list(csv.DictReader(f))
    assert len(rows) >= 1
    assert float(rows[-1]["coverage_frac"]) > 0
    d = np.load(pos_path)
    assert len(d["x"]) == 129 and len(d["z"]) == 129  # 1 initial + 128 steps
