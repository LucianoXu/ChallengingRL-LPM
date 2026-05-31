"""Grid runner for the maze exploration comparison.

Usage: python run_grid.py --steps 20000 --seeds 1 2  [--dry-run]

Runs on CPU by default: the decoder-based methods (LPM/MSE) are faster on CPU
than MPS, where ConvTranspose falls back off-device.
"""
import argparse
import itertools
import os
import subprocess

EXP = os.path.dirname(os.path.abspath(__file__))
PY = os.path.join(os.path.dirname(os.path.dirname(EXP)), ".venv", "bin", "python")
RESULTS = os.path.join(EXP, "results")
POSITIONS = os.path.join(EXP, "positions")

METHODS = ["lpm", "rnd", "icm", "mse", "none"]
VARIANTS = ["nonoise", "noisy_tv", "action_noise"]


def enough_rows(path, n=2):
    if not os.path.exists(path):
        return False
    with open(path) as f:
        return sum(1 for _ in f) > n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2])
    ap.add_argument("--methods", nargs="+", default=METHODS)
    ap.add_argument("--variants", nargs="+", default=VARIANTS)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--obs-scale", type=float, default=1.0)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    os.makedirs(RESULTS, exist_ok=True); os.makedirs(POSITIONS, exist_ok=True)
    os.makedirs("/tmp/maze_logs", exist_ok=True)
    env = dict(os.environ, PYTHONPATH=EXP, PYTORCH_ENABLE_MPS_FALLBACK="1")

    runs = list(itertools.product(args.methods, args.variants, args.seeds))
    print(f"{len(runs)} runs planned")
    for method, variant, seed in runs:
        rid = f"{method}-{variant}-s{seed}"
        csv_path = os.path.join(RESULTS, rid + ".csv")
        if enough_rows(csv_path):
            print(f"[skip] {rid} (already has results)"); continue
        cmd = [PY, os.path.join(EXP, "train_maze.py"), "--method", method,
               "--variant", variant, "--seed", str(seed), "--steps", str(args.steps),
               "--device", args.device, "--obs-scale", str(args.obs_scale),
               "--csv-log", csv_path,
               "--pos-log", os.path.join(POSITIONS, rid + ".npz"),
               "--log-interval", "5"]
        print(f"[run] {rid}: {' '.join(cmd)}")
        if args.dry_run:
            continue
        with open(f"/tmp/maze_logs/{rid}.out", "w") as out:
            res = subprocess.run(cmd, env=env, cwd=EXP, stdout=out, stderr=subprocess.STDOUT)
        if res.returncode != 0:
            print(f"[FAIL] {rid} (see /tmp/maze_logs/{rid}.out)")


if __name__ == "__main__":
    main()
