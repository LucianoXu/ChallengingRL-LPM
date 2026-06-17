"""Grid runner for the maze exploration comparison.

Usage:
  python run_grid.py --steps 20000 --seeds 1 2                  # sequential
  python run_grid.py --steps 20000 --seeds 1 2 --jobs 8         # 8 runs in parallel
  python run_grid.py --dry-run

Runs on CPU by default: the decoder-based methods (LPM/MSE) are faster on CPU
than MPS, where ConvTranspose falls back off-device.

Parallelism: the 30 runs are independent, so `--jobs N` launches up to N runs
concurrently. Each worker's CPU-thread count is pinned (`--threads-per-job`,
default chosen so jobs*threads ~= core count) because a single training step is
a batch-size-1 forward pass that does not benefit from many intra-op threads;
the win comes from running many processes, not many threads per process.
"""
import argparse
import itertools
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed

EXP = os.path.dirname(os.path.abspath(__file__))
PY = os.path.join(os.path.dirname(os.path.dirname(EXP)), ".venv", "bin", "python")
# Experiment artifacts live under <repo>/expr_data/ (gitignored), not in the
# package tree. EXP -> Miniworld -> LPM_exploration -> repo root.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(EXP)))
EXPR_DATA = os.path.join(REPO_ROOT, "expr_data", "miniworld")
RESULTS = os.path.join(EXPR_DATA, "results")
POSITIONS = os.path.join(EXPR_DATA, "positions")

METHODS = ["lpm", "rnd", "icm", "mse", "none"]
VARIANTS = ["nonoise", "noisy_tv", "action_noise"]


def is_complete(rid):
    """A run is complete iff its position .npz exists (written only at the end)."""
    return os.path.exists(os.path.join(POSITIONS, rid + ".npz"))


def run_one(rid, cmd, base_env, threads):
    env = dict(base_env)
    # Pin per-worker CPU threads so N parallel workers don't oversubscribe cores.
    for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
              "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        env[k] = str(threads)
    with open(f"/tmp/maze_logs/{rid}.out", "w") as out:
        res = subprocess.run(cmd, env=env, cwd=EXP, stdout=out, stderr=subprocess.STDOUT)
    return rid, res.returncode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2])
    ap.add_argument("--methods", nargs="+", default=METHODS)
    ap.add_argument("--variants", nargs="+", default=VARIANTS)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--obs-scale", type=float, default=1.0)
    ap.add_argument("--jobs", type=int, default=1, help="concurrent runs (independent)")
    ap.add_argument("--threads-per-job", type=int, default=0,
                    help="CPU threads per worker; 0 = auto (cores // jobs, min 1)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    os.makedirs(RESULTS, exist_ok=True); os.makedirs(POSITIONS, exist_ok=True)
    os.makedirs("/tmp/maze_logs", exist_ok=True)
    base_env = dict(os.environ, PYTHONPATH=EXP, PYTORCH_ENABLE_MPS_FALLBACK="1")

    cores = os.cpu_count() or 4
    threads = args.threads_per_job or max(1, cores // max(1, args.jobs))

    runs = list(itertools.product(args.methods, args.variants, args.seeds))
    pending = []
    for method, variant, seed in runs:
        rid = f"{method}-{variant}-s{seed}"
        if is_complete(rid):
            print(f"[skip] {rid} (already complete)"); continue
        # "uniform" = the no-learning random-walk control: train_maze has no such
        # --method, so map it to `--method none --random-policy`.
        train_method = "none" if method == "uniform" else method
        cmd = [PY, os.path.join(EXP, "train_maze.py"), "--method", train_method,
               "--variant", variant, "--seed", str(seed), "--steps", str(args.steps),
               "--device", args.device, "--obs-scale", str(args.obs_scale),
               "--csv-log", os.path.join(RESULTS, rid + ".csv"),
               "--pos-log", os.path.join(POSITIONS, rid + ".npz"),
               "--log-interval", "5"]
        if method == "uniform":
            cmd.append("--random-policy")
        pending.append((rid, cmd))

    print(f"{len(runs)} runs total, {len(pending)} to run, "
          f"jobs={args.jobs}, threads/job={threads} (cores={cores})")
    if args.dry_run:
        for rid, cmd in pending:
            print(f"[run] {rid}")
        return

    if args.jobs <= 1:
        for rid, cmd in pending:
            print(f"[run] {rid}")
            _, rc = run_one(rid, cmd, base_env, threads)
            if rc != 0:
                print(f"[FAIL] {rid} (see /tmp/maze_logs/{rid}.out)")
        return

    with ThreadPoolExecutor(max_workers=args.jobs) as ex:
        futs = {ex.submit(run_one, rid, cmd, base_env, threads): rid
                for rid, cmd in pending}
        for fut in as_completed(futs):
            rid, rc = fut.result()
            tag = "done" if rc == 0 else "FAIL"
            print(f"[{tag}] {rid}" + ("" if rc == 0 else f" (see /tmp/maze_logs/{rid}.out)"))


if __name__ == "__main__":
    main()
