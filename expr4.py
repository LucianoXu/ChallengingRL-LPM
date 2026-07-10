"""MiniGrid beta and observation-noise sweeps.

Dry-run preview by default:

    python expr4.py

Launch the sweeps:

    python expr4.py --run --python ./LPM_exploration/.venv/bin/python

Included sweeps:

1. Beta sweep on FourRooms and MultiRoom-N6 for RND/LPM/ICM:
   beta in {0, 0.0005, 0.001, 0.005, 0.01, 0.05}

2. FourRooms observation-noise sweep for none/entropy/RND/LPM/ICM:
   noise_prob in {0, 0.01, 0.02, ..., 0.10}

Training is chunked and resumable via the same progress sidecars used by
`minigrid_exp/run_grid.py`.
"""
from __future__ import annotations

import argparse
import math
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parent
EXP = ROOT / "minigrid_exp"
RUNNER = EXP / "run_grid.py"
ANALYZER = EXP / "analyze.py"
MODEL_DIR = ROOT / "expr_data" / "minigrid" / "results" / "models" / "ppo"

FOURROOMS = "MiniGrid-FourRooms-v0"
MULTIROOM = "MiniGrid-MultiRoom-N6-v0"

DEFAULT_BETAS = [0.0, 0.0005, 0.001, 0.005, 0.01, 0.05]
DEFAULT_NOISE_PROBS = [i / 100 for i in range(0, 11)]
DEFAULT_METHODS = ["rnd", "lpm", "icm"]
BASELINE_METHODS = ["none", "entropy"]
NOISE_VARIANTS = {"baseline_noise", "intrinsic_noise"}


@dataclass(frozen=True)
class SweepSpec:
    name: str
    env: str
    steps: int
    variants: tuple[str, ...]
    methods: tuple[str, ...]
    betas: tuple[float, ...] | None
    noise_probs: tuple[float, ...]


def _venv_python() -> str:
    linux = ROOT / "LPM_exploration" / ".venv" / "bin" / "python"
    windows = ROOT / "LPM_exploration" / ".venv" / "Scripts" / "python.exe"
    if linux.exists():
        return str(linux)
    if windows.exists():
        return str(windows)
    return sys.executable


def _tag(beta: float | None, noise: bool, noise_prob: float) -> str | None:
    parts = []
    if beta is not None:
        parts.append(f"beta{beta:g}")
    if noise:
        parts.append(f"np{noise_prob:g}")
    return "_".join(parts) or None


def _run_name(env_id: str, variant: str, method: str, seed: int,
              beta: float | None, noise_prob: float) -> str:
    tag = _tag(beta, variant in NOISE_VARIANTS, noise_prob)
    suffix = f"__{tag}" if tag else ""
    return f"{env_id}__{variant}__{method}__seed_{seed}{suffix}".replace("/", "_")


def _progress(run_name: str) -> int:
    sidecar = MODEL_DIR / f"{run_name}.progress"
    if not sidecar.exists():
        return 0
    try:
        return int(sidecar.read_text().strip())
    except (OSError, ValueError):
        return 0


def _specs(args: argparse.Namespace) -> list[SweepSpec]:
    specs = []
    enabled = set(args.sweeps)
    if "beta" in enabled:
        specs.extend([
            SweepSpec(
                name="beta_fourrooms",
                env=FOURROOMS,
                steps=args.fourrooms_steps,
                variants=("intrinsic_no_noise",),
                methods=tuple(args.methods),
                betas=tuple(args.beta_values),
                noise_probs=(0.10,),
            ),
            SweepSpec(
                name="beta_multiroom",
                env=MULTIROOM,
                steps=args.multiroom_steps,
                variants=("intrinsic_no_noise",),
                methods=tuple(args.methods),
                betas=tuple(args.beta_values),
                noise_probs=(0.10,),
            ),
        ])
    if "noise" in enabled:
        specs.append(
            SweepSpec(
                name="noise_fourrooms",
                env=FOURROOMS,
                steps=args.fourrooms_steps,
                variants=("baseline_noise", "intrinsic_noise"),
                methods=tuple(args.methods),
                betas=tuple(args.noise_betas) if args.noise_betas else None,
                noise_probs=tuple(args.noise_probs),
            )
        )
    return specs


def _expected_cells(specs: list[SweepSpec], seeds: list[int]) -> list[dict]:
    cells = []
    for spec in specs:
        for variant in spec.variants:
            intrinsic = variant.startswith("intrinsic")
            methods = spec.methods if intrinsic else tuple(BASELINE_METHODS)
            betas = spec.betas if intrinsic and spec.betas is not None else (None,)
            noise_probs = spec.noise_probs if variant in NOISE_VARIANTS else (0.10,)
            for method in methods:
                for beta in betas:
                    for noise_prob in noise_probs:
                        for seed in seeds:
                            run_name = _run_name(spec.env, variant, method, seed, beta, noise_prob)
                            cells.append({
                                "spec": spec.name,
                                "env": spec.env,
                                "variant": variant,
                                "method": method,
                                "beta": beta,
                                "noise_prob": noise_prob,
                                "seed": seed,
                                "steps": spec.steps,
                                "run_name": run_name,
                            })
    return cells


def _incomplete(cells: list[dict]) -> list[dict]:
    return [cell for cell in cells if _progress(cell["run_name"]) < cell["steps"]]


def _runner_cmd(spec: SweepSpec, args: argparse.Namespace, dry_run: bool) -> list[str]:
    py = args.python_path or _venv_python()
    cmd = [
        py,
        str(RUNNER),
        "--envs", spec.env,
        "--variants", *spec.variants,
        "--steps", str(spec.steps),
        "--chunk-steps", str(args.chunk_steps),
        "--seeds", *[str(seed) for seed in args.seeds],
        "--methods", *spec.methods,
        "--noise-probs", *[f"{p:g}" for p in spec.noise_probs],
        "--jobs", str(args.jobs),
        "--threads-per-job", str(args.threads_per_job),
        "--python", py,
    ]
    if spec.betas is not None:
        cmd += ["--betas", *[f"{b:g}" for b in spec.betas]]
    if dry_run:
        cmd.append("--dry-run")
    return cmd


def _base_env(args: argparse.Namespace) -> dict[str, str]:
    py = args.python_path or _venv_python()
    env = dict(os.environ)
    env["PYTHONPATH"] = str(EXP) + os.pathsep + env.get("PYTHONPATH", "")
    env["MINIGRID_PYTHON"] = py
    env.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    return env


def _run_cmd(cmd: list[str], args: argparse.Namespace) -> int:
    print(" ".join(cmd), flush=True)
    res = subprocess.run(cmd, cwd=str(ROOT), env=_base_env(args))
    return int(res.returncode)


def _analyze(args: argparse.Namespace) -> int:
    py = args.python_path or _venv_python()
    cmd = [py, str(ANALYZER)]
    print(f"[expr4] analyze: {' '.join(cmd)}", flush=True)
    res = subprocess.run(cmd, cwd=str(ROOT), env=_base_env(args))
    return int(res.returncode)


def _print_plan(specs: list[SweepSpec], cells: list[dict], args: argparse.Namespace) -> None:
    pending = _incomplete(cells)
    print("[expr4] MiniGrid hyperparameter sweeps")
    print(f"[expr4] sweeps: {', '.join(args.sweeps)}")
    print(f"[expr4] methods: {', '.join(args.methods)}")
    print(f"[expr4] seeds: {', '.join(str(s) for s in args.seeds)}")
    print(f"[expr4] beta values: {', '.join(f'{b:g}' for b in args.beta_values)}")
    print(f"[expr4] FourRooms noise probs: {', '.join(f'{p:g}' for p in args.noise_probs)}")
    print("[expr4] specs:")
    for spec in specs:
        betas = "default" if spec.betas is None else " ".join(f"{b:g}" for b in spec.betas)
        probs = " ".join(f"{p:g}" for p in spec.noise_probs)
        spec_count = sum(1 for cell in cells if cell["spec"] == spec.name)
        spec_pending = sum(1 for cell in pending if cell["spec"] == spec.name)
        print(
            "  "
            f"{spec.name}: env={spec.env}, steps={spec.steps:,}, "
            f"variants={','.join(spec.variants)}, betas={betas}, "
            f"noise_probs={probs}, cells={spec_count}, pending={spec_pending}"
        )
    print(f"[expr4] cells total: {len(cells)}, pending: {len(pending)}")
    if pending:
        print("[expr4] first pending cells:")
        for cell in pending[: args.preview_cells]:
            done = _progress(cell["run_name"])
            print(f"  {cell['run_name']} ({done:,}/{cell['steps']:,})")
        if len(pending) > args.preview_cells:
            print(f"  ... {len(pending) - args.preview_cells} more")


def main() -> int:
    ap = argparse.ArgumentParser(description="Run MiniGrid beta and FourRooms noise sweeps.")
    ap.add_argument("--run", action="store_true",
                    help="actually launch training; default is dry-run preview")
    ap.add_argument("--analyze-only", action="store_true",
                    help="skip training and only aggregate existing eval outputs")
    ap.add_argument("--no-analyze", action="store_true",
                    help="do not run analyze.py after completion")
    ap.add_argument("--python", dest="python_path",
                    help="Python executable for run_grid.py, train_one.py, and analyze.py")
    ap.add_argument("--sweeps", nargs="+", choices=["beta", "noise"],
                    default=["beta", "noise"])
    ap.add_argument("--methods", nargs="+", default=list(DEFAULT_METHODS),
                    choices=["rnd", "lpm", "icm"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--beta-values", type=float, nargs="+", default=list(DEFAULT_BETAS))
    ap.add_argument("--noise-probs", type=float, nargs="+", default=list(DEFAULT_NOISE_PROBS))
    ap.add_argument("--noise-betas", type=float, nargs="+",
                    help="optional beta sweep inside the FourRooms noise-prob sweep; "
                         "default uses each method's configured beta")
    ap.add_argument("--fourrooms-steps", type=int, default=2_000_000)
    ap.add_argument("--multiroom-steps", type=int, default=3_000_000)
    ap.add_argument("--chunk-steps", type=int, default=300_000)
    ap.add_argument("--rounds", type=int, default=0,
                    help="max chunk rounds; 0 = enough rounds for the largest requested budget")
    ap.add_argument("--jobs", type=int, default=16)
    ap.add_argument("--threads-per-job", type=int, default=1)
    ap.add_argument("--preview-cells", type=int, default=14)
    ap.add_argument("--print-run-grid-dry-run", action="store_true",
                    help="also call run_grid.py --dry-run for each selected sweep spec")
    args = ap.parse_args()

    if args.analyze_only:
        return _analyze(args)

    specs = _specs(args)
    cells = _expected_cells(specs, args.seeds)
    _print_plan(specs, cells, args)

    if not args.run:
        if args.print_run_grid_dry_run:
            for spec in specs:
                rc = _run_cmd(_runner_cmd(spec, args, dry_run=True), args)
                if rc != 0:
                    return rc
        return 0

    max_rounds = args.rounds or max(math.ceil(spec.steps / args.chunk_steps) for spec in specs)
    for round_idx in range(1, max_rounds + 1):
        pending = _incomplete(cells)
        if not pending:
            print("[expr4] all cells complete")
            break

        pending_specs = {cell["spec"] for cell in pending}
        print(f"[expr4] round {round_idx}/{max_rounds}: {len(pending)} pending cells")
        for spec in specs:
            if spec.name not in pending_specs:
                continue
            rc = _run_cmd(_runner_cmd(spec, args, dry_run=False), args)
            if rc != 0:
                return rc

    pending = _incomplete(cells)
    if pending:
        print(f"[expr4] stopped with {len(pending)} incomplete cells")
        print("[expr4] rerun the same command later to resume")
        return 1

    if args.no_analyze:
        return 0
    return _analyze(args)


if __name__ == "__main__":
    raise SystemExit(main())
