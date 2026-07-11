"""Process-parallel grid runner for the MiniGrid intrinsic-reward experiments.

One subprocess per (env, variant, method, beta, seed) cell -- process-level
parallelism (the GIL makes thread pools poor for SB3 stepping). Resumable: a
cell is complete iff its progress sidecar records >= total requested steps.

Usage:
  PYTHONPATH=. python run_grid.py --steps 1000000 --jobs 16
  PYTHONPATH=. python run_grid.py --betas 0.0 0.01 0.05 0.1 0.5 --jobs 16   # beta sweep
  PYTHONPATH=. python run_grid.py --dry-run
"""
import argparse
import itertools
import os
import subprocess
import sys
# ThreadPoolExecutor (not ProcessPoolExecutor): run_cell only blocks on
# subprocess.run, so threads give identical parallelism (the real work is in the
# child processes) without multiprocessing's fork/resource-tracker fragility,
# which fails to start workers in detached/background (non-TTY) contexts.
from concurrent.futures import ThreadPoolExecutor, as_completed

import config

EXP = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_PY = os.path.join(os.path.dirname(EXP), "LPM_exploration", ".venv", "bin", "python")
PY = os.environ.get("MINIGRID_PYTHON", _DEFAULT_PY if os.path.exists(_DEFAULT_PY) else sys.executable)

# (variant_name, intrinsic, noise)
VARIANTS = [(v["name"], v["intrinsic"], v["noise"]) for v in config.VARIANTS]
LOG_DIR = os.path.join(config.RESULTS_DIR, "runner_logs")


def cell_complete(env_id, variant, method, seed, tag, total_steps):
    """Return True iff the progress sidecar shows >= total_steps completed."""
    suffix = f"__{tag}" if tag else ""
    run_name = f"{env_id}__{variant}__{method}__seed_{seed}{suffix}".replace("/", "_")
    sidecar = os.path.join(config.MODELS_DIR, f"{run_name}.progress")
    if not os.path.exists(sidecar):
        return False
    try:
        progress = int(open(sidecar).read().strip())
        return progress >= total_steps
    except (ValueError, OSError):
        return False


def run_cell(cmd, logfile, threads):
    # Pin each run's CPU-thread count so jobs*threads saturates the box without
    # oversubscribing. torch/numpy read these at import.
    tvars = {k: str(threads) for k in
             ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
              "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")}
    with open(logfile, "w") as fh:
        return subprocess.run(cmd, cwd=EXP, stdout=fh, stderr=subprocess.STDOUT,
                              env={**os.environ, "PYTHONPATH": EXP, **tvars}).returncode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=config.TOTAL_TIMESTEPS)
    ap.add_argument("--chunk-steps", type=int, default=config.CHUNK_STEPS,
                    help="max timesteps per chunk (default: config.CHUNK_STEPS)")
    ap.add_argument("--seeds", type=int, nargs="+", default=config.SEEDS)
    ap.add_argument("--methods", nargs="+", default=["rnd", "lpm", "icm"])
    ap.add_argument("--betas", type=float, nargs="+", default=[None],
                    help="intrinsic-reward scales to sweep; None = config default")
    ap.add_argument("--noise-probs", type=float, nargs="+", default=[0.10],
                    help="observation noise probabilities to sweep (noise variants only)")
    ap.add_argument("--jobs", type=int, default=16)
    ap.add_argument("--threads-per-job", type=int, default=0,
                    help="CPU threads per run (OMP/MKL/...); 0 = auto (cores // jobs, min 1) to saturate the box")
    ap.add_argument("--python", default=None,
                    help="Python executable used for train_one.py children")
    ap.add_argument("--envs", nargs="+", default=None,
                    help="restrict to these env ids; default = all config envs")
    ap.add_argument("--variants", nargs="+", default=None,
                    help="restrict to these variant names; default = all variants")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    global PY
    if a.python:
        PY = a.python

    os.makedirs(LOG_DIR, exist_ok=True)
    envs = a.envs if a.envs else [e for tier in config.ENVIRONMENTS.values() for e in tier]

    variants = VARIANTS
    if a.variants:
        variants = [(name, intr, noise) for name, intr, noise in VARIANTS if name in a.variants]

    pending = []
    for env_id, (variant, intrinsic, noise), seed in itertools.product(envs, variants, a.seeds):
        # Non-intrinsic baselines run once per method-agnostic cell; tag by "none".
        methods = a.methods if intrinsic else ["none", "entropy"]
        betas = a.betas if intrinsic else [None]
        noise_probs = a.noise_probs if noise else [0.10]
        for method, beta, noise_prob in itertools.product(methods, betas, noise_probs):
            parts = []
            if beta is not None:
                parts.append(f"beta{beta:g}")
            if noise:
                parts.append(f"np{noise_prob:g}")
            tag = "_".join(parts) or None
            if cell_complete(env_id, variant, method, seed, tag, a.steps):
                continue
            cmd = [PY, os.path.join(EXP, "train_one.py"), "--env", env_id,
                   "--method", method, "--seed", str(seed), "--steps", str(a.steps),
                   "--chunk-steps", str(a.chunk_steps)]
            if intrinsic:
                cmd.append("--intrinsic")
            if noise:
                cmd.append("--noise")
            if beta is not None:
                cmd += ["--beta", str(beta)]
            cmd += ["--noise-prob", str(noise_prob)]
            rid = f"{env_id}__{variant}__{method}__s{seed}" + (f"__{tag}" if tag else "")
            pending.append((rid.replace("/", "_"), cmd))

    # Default to 1 thread/job: proven clean (the 72-run OMP=1 grid had 0 failures),
    # and cranking OMP threads neither sped runs up (375 vs 414 fps) nor saturated
    # the box productively -- it correlated with SIGKILLs of the heavier runs.
    # Opt in to more via --threads-per-job only after validating memory safety.
    cores = os.cpu_count() or 8
    threads = a.threads_per_job or 1
    print(f"{len(pending)} cells to run, jobs={a.jobs}, threads/job={threads} (cores={cores})")
    if a.dry_run:
        for rid, _ in pending:
            print("[run]", rid)
        return

    done = failed = 0
    with ThreadPoolExecutor(max_workers=a.jobs) as ex:
        futs = {ex.submit(run_cell, cmd, os.path.join(LOG_DIR, f"{rid}.out"), threads): rid
                for rid, cmd in pending}
        for fut in as_completed(futs):
            rid = futs[fut]
            try:
                rc = fut.result()
            except Exception as exc:  # don't let one cell abort the whole grid
                rc, exc_note = 1, f" -- exception: {exc}"
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
