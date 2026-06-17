"""Process-parallel grid runner for the MiniGrid intrinsic-reward experiments.

One subprocess per (env, variant, method, beta, seed) cell — process-level
parallelism (the GIL makes thread pools poor for SB3 stepping). Resumable: a
cell is complete iff its saved model .zip exists under expr_data/minigrid.

Usage:
  PYTHONPATH=. python run_grid.py --steps 1000000 --jobs 32
  PYTHONPATH=. python run_grid.py --betas 0.0 0.01 0.05 0.1 0.5 --jobs 32   # beta sweep
  PYTHONPATH=. python run_grid.py --dry-run
"""
import argparse
import itertools
import os
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed

import config

EXP = os.path.dirname(os.path.abspath(__file__))
PY = os.path.join(os.path.dirname(EXP), "LPM_exploration", ".venv", "bin", "python")

# (variant_name, intrinsic, noise)
VARIANTS = [(v["name"], v["intrinsic"], v["noise"]) for v in config.VARIANTS]


def cell_complete(env_id, variant, method, seed, tag):
    suffix = f"__{tag}" if tag else ""
    run_name = f"{env_id}__{variant}__{method}__seed_{seed}{suffix}".replace("/", "_")
    return os.path.exists(os.path.join(config.MODELS_DIR, f"{run_name}.zip"))


def run_cell(cmd, logfile):
    with open(logfile, "w") as fh:
        return subprocess.run(cmd, cwd=EXP, stdout=fh, stderr=subprocess.STDOUT,
                              env={**os.environ, "PYTHONPATH": EXP,
                                   "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}).returncode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=config.TOTAL_TIMESTEPS)
    ap.add_argument("--seeds", type=int, nargs="+", default=config.SEEDS)
    ap.add_argument("--methods", nargs="+", default=["rnd", "lpm", "count"])
    ap.add_argument("--betas", type=float, nargs="+", default=[None],
                    help="intrinsic-reward scales to sweep; None = config default")
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--envs", nargs="+", default=None,
                    help="restrict to these env ids; default = all config envs")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    os.makedirs("/tmp/minigrid_logs", exist_ok=True)
    envs = a.envs if a.envs else [e for tier in config.ENVIRONMENTS.values() for e in tier]

    pending = []
    for env_id, (variant, intrinsic, noise), seed in itertools.product(envs, VARIANTS, a.seeds):
        # Non-intrinsic baselines run once per method-agnostic cell; tag by "none".
        methods = a.methods if intrinsic else ["none"]
        betas = a.betas if intrinsic else [None]
        for method, beta in itertools.product(methods, betas):
            tag = None if beta is None else f"beta{beta:g}"
            if cell_complete(env_id, variant, method, seed, tag):
                continue
            cmd = [PY, os.path.join(EXP, "train_one.py"), "--env", env_id,
                   "--method", method, "--seed", str(seed), "--steps", str(a.steps)]
            if intrinsic:
                cmd.append("--intrinsic")
            if noise:
                cmd.append("--noise")
            if beta is not None:
                cmd += ["--beta", str(beta)]
            rid = f"{env_id}__{variant}__{method}__s{seed}" + (f"__{tag}" if tag else "")
            pending.append((rid.replace("/", "_"), cmd))

    print(f"{len(pending)} cells to run, jobs={a.jobs}")
    if a.dry_run:
        for rid, _ in pending:
            print("[run]", rid)
        return

    done = failed = 0
    with ProcessPoolExecutor(max_workers=a.jobs) as ex:
        futs = {ex.submit(run_cell, cmd, f"/tmp/minigrid_logs/{rid}.out"): rid
                for rid, cmd in pending}
        for fut in as_completed(futs):
            rid = futs[fut]
            try:
                rc = fut.result()
            except Exception as exc:  # don't let one cell abort the whole grid
                rc, exc_note = 1, f" — exception: {exc}"
            else:
                exc_note = ""
            if rc == 0:
                done += 1
            else:
                failed += 1
            print(f"[{'done' if rc == 0 else 'FAIL'}] {rid}{exc_note}", flush=True)
    print(f"=== grid finished: {done} done, {failed} failed, "
          f"{len(pending)} attempted ===", flush=True)


if __name__ == "__main__":
    main()
