"""Bootstrap script for MiniWorld LPM hyperparameter experiment 1.

Run a dry-run preview by default:

    python expr1.py

Launch the actual sweep:

    python expr1.py --run
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
RUNNER = ROOT / "LPM_exploration" / "Miniworld" / "experiments" / "run_hparam_sweep.py"
DEFAULT_SWEEP_NAME = "expr1_lpm_core_action_noise"


def _venv_python() -> str:
    linux = ROOT / "LPM_exploration" / ".venv" / "bin" / "python"
    windows = ROOT / "LPM_exploration" / ".venv" / "Scripts" / "python.exe"
    if linux.exists():
        return str(linux)
    if windows.exists():
        return str(windows)
    return sys.executable


def build_cmd(args: argparse.Namespace) -> list[str]:
    cmd = [
        _venv_python(),
        str(RUNNER),
        "--preset", "lpm_core",
        "--sweep-name", args.sweep_name,
        "--jobs", str(args.jobs),
        "--threads-per-job", str(args.threads_per_job),
    ]
    if not args.run:
        cmd.append("--dry-run")
    if args.force:
        cmd.append("--force")
    if args.max_runs:
        cmd += ["--max-runs", str(args.max_runs)]
    if args.steps is not None:
        cmd += ["--steps", str(args.steps)]
    if args.seeds:
        cmd += ["--seeds", *[str(seed) for seed in args.seeds]]
    if args.lambda_values:
        cmd += ["--lambda-intrinsic-values", *[str(v) for v in args.lambda_values]]
    if args.entropy_values:
        cmd += ["--entropy-coef-values", *[str(v) for v in args.entropy_values]]
    if args.reward_spaces:
        cmd += ["--lpm-reward-space-values", *args.reward_spaces]
    return cmd


def main() -> int:
    ap = argparse.ArgumentParser(description="Bootstrap MiniWorld LPM sweep experiment 1.")
    ap.add_argument("--run", action="store_true",
                    help="actually launch the sweep; default is dry-run preview")
    ap.add_argument("--force", action="store_true",
                    help="rerun cells that already have completed position logs")
    ap.add_argument("--sweep-name", default=DEFAULT_SWEEP_NAME)
    ap.add_argument("--jobs", type=int, default=96)
    ap.add_argument("--threads-per-job", type=int, default=1)
    ap.add_argument("--max-runs", type=int, default=0,
                    help="debug cap; useful together with --run for a smoke launch")
    ap.add_argument("--steps", type=int,
                    help="override training steps from the lpm_core preset")
    ap.add_argument("--seeds", type=int, nargs="+",
                    help="override seeds from the lpm_core preset")
    ap.add_argument("--lambda-values", type=float, nargs="+",
                    help="override intrinsic reward scale values")
    ap.add_argument("--entropy-values", type=float, nargs="+",
                    help="override entropy coefficient values")
    ap.add_argument("--reward-spaces", choices=["log", "raw"], nargs="+",
                    help="override LPM reward-space values")
    args = ap.parse_args()

    cmd = build_cmd(args)
    mode = "RUN" if args.run else "DRY-RUN"
    print(f"[expr1] {mode}: {' '.join(cmd)}", flush=True)
    env = dict(os.environ)
    env["PYTHONPATH"] = str(RUNNER.parent) + os.pathsep + env.get("PYTHONPATH", "")
    env.setdefault("PYGLET_HEADLESS", "true")
    res = subprocess.run(cmd, cwd=str(ROOT), env=env)
    return int(res.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
