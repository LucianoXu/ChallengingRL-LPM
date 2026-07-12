"""Run the MiniGrid full exploration matrix.

Dry-run preview by default:

    python expr3.py

Launch training:

    python expr3.py --run

This is a top-level convenience launcher for:

- 3 environments: DoorKey-5x5, FourRooms, MultiRoom-N6
- 2 noise settings: clean and noisy observation variants
- 5 visible methods: none, entropy, RND, LPM, ICM

The underlying `minigrid_exp/run_grid.py` treats `none` and `entropy` as the
baseline methods, and treats `rnd`, `lpm`, and `icm` as intrinsic methods.
Training is chunked and resumable. Each worker commits PPO and intrinsic state
together, and repeated rounds restore that complete learning state before the
next chunk.
"""
from __future__ import annotations

import argparse
import math
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
EXP = ROOT / "minigrid_exp"
RUNNER = EXP / "run_grid.py"
ANALYZER = EXP / "analyze.py"
DEFAULT_OUTPUT_DIR = ROOT / "expr_data" / "minigrid" / "exp3_final"

ENV_BUDGETS = {
    "MiniGrid-DoorKey-5x5-v0": 1_000_000,
    "MiniGrid-FourRooms-v0": 2_000_000,
    "MiniGrid-MultiRoom-N6-v0": 3_000_000,
}
DEFAULT_VARIANTS = [
    "baseline_no_noise",
    "intrinsic_no_noise",
    "baseline_noise",
    "intrinsic_noise",
]
BASELINE_METHODS = ["none", "entropy"]
INTRINSIC_METHODS = ["rnd", "lpm", "icm"]
NOISE_VARIANTS = {"baseline_noise", "intrinsic_noise"}


def _load_minigrid_config(args: argparse.Namespace):
    os.environ["MINIGRID_EXPR_DATA"] = str(_output_dir(args))
    sys.path.insert(0, str(EXP))
    import config  # noqa: WPS433
    return config


def _venv_python() -> str:
    linux = ROOT / "LPM_exploration" / ".venv" / "bin" / "python"
    windows = ROOT / "LPM_exploration" / ".venv" / "Scripts" / "python.exe"
    if linux.exists():
        return str(linux)
    if windows.exists():
        return str(windows)
    return sys.executable


def _env_steps(args: argparse.Namespace) -> dict[str, int]:
    envs = list(args.envs or ENV_BUDGETS)
    steps = {
        "MiniGrid-DoorKey-5x5-v0": args.doorkey_steps,
        "MiniGrid-FourRooms-v0": args.fourrooms_steps,
        "MiniGrid-MultiRoom-N6-v0": args.multiroom_steps,
    }
    if args.steps is not None:
        steps = {env: args.steps for env in steps}
    return {env: steps[env] for env in envs}


def _tag(beta: float | None, noise: bool, noise_prob: float) -> str | None:
    parts = []
    if beta is not None:
        parts.append(f"beta{beta:g}")
    if noise:
        parts.append(f"np{noise_prob:g}")
    return "_".join(parts) or None


def _run_name(env_id: str, variant: str, method: str, seed: int,
              beta: float | None, noise_prob: float) -> str:
    noise = variant in NOISE_VARIANTS
    tag = _tag(beta, noise, noise_prob)
    suffix = f"__{tag}" if tag else ""
    return f"{env_id}__{variant}__{method}__seed_{seed}{suffix}".replace("/", "_")


def _output_dir(args: argparse.Namespace) -> Path:
    return Path(args.output_dir).expanduser().resolve()


def _progress(run_name: str, args: argparse.Namespace) -> int:
    sidecar = _output_dir(args) / "results" / "models" / "ppo" / f"{run_name}.progress"
    if not sidecar.exists():
        return 0
    try:
        return int(sidecar.read_text().strip())
    except (OSError, ValueError):
        return 0


def _expected_cells(args: argparse.Namespace, env_steps: dict[str, int]) -> list[dict]:
    cells = []
    variants = args.variants or DEFAULT_VARIANTS
    betas: list[float | None] = args.betas if args.betas else [None]
    for env_id, steps in env_steps.items():
        for variant in variants:
            intrinsic = variant.startswith("intrinsic")
            methods = args.methods if intrinsic else BASELINE_METHODS
            cell_betas = betas if intrinsic else [None]
            noise_probs = args.noise_probs if variant in NOISE_VARIANTS else [0.10]
            for method in methods:
                for beta in cell_betas:
                    for noise_prob in noise_probs:
                        for seed in args.seeds:
                            run_name = _run_name(env_id, variant, method, seed, beta, noise_prob)
                            cells.append({
                                "env": env_id,
                                "steps": steps,
                                "variant": variant,
                                "method": method,
                                "seed": seed,
                                "beta": beta,
                                "noise_prob": noise_prob,
                                "run_name": run_name,
                            })
    return cells


def _incomplete(cells: list[dict], args: argparse.Namespace) -> list[dict]:
    return [
        cell for cell in cells
        if _progress(cell["run_name"], args) < cell["steps"]
    ]


def _runner_cmd(args: argparse.Namespace, env_id: str, steps: int, dry_run: bool) -> list[str]:
    py = args.python_path or _venv_python()
    cmd = [
        py,
        str(RUNNER),
        "--envs", env_id,
        "--steps", str(steps),
        "--chunk-steps", str(args.chunk_steps),
        "--seeds", *[str(seed) for seed in args.seeds],
        "--methods", *args.methods,
        "--noise-probs", *[str(p) for p in args.noise_probs],
        "--jobs", str(args.jobs),
        "--threads-per-job", str(args.threads_per_job),
        "--python", py,
    ]
    if args.variants:
        cmd += ["--variants", *args.variants]
    if args.betas:
        cmd += ["--betas", *[str(beta) for beta in args.betas]]
    if dry_run:
        cmd.append("--dry-run")
    return cmd


def _base_env(args: argparse.Namespace) -> dict[str, str]:
    py = args.python_path or _venv_python()
    env = dict(os.environ)
    env["PYTHONPATH"] = str(EXP) + os.pathsep + env.get("PYTHONPATH", "")
    env["MINIGRID_PYTHON"] = py
    env["MINIGRID_EXPR_DATA"] = str(_output_dir(args))
    env.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    return env


def _run_cmd(cmd: list[str], args: argparse.Namespace) -> int:
    print(" ".join(cmd), flush=True)
    res = subprocess.run(cmd, cwd=str(ROOT), env=_base_env(args))
    return int(res.returncode)


def _analyze(args: argparse.Namespace) -> int:
    py = args.python_path or _venv_python()
    cmd = [py, str(ANALYZER)]
    print(f"[expr3] analyze: {' '.join(cmd)}", flush=True)
    res = subprocess.run(cmd, cwd=str(ROOT), env=_base_env(args))
    return int(res.returncode)


def _preflight(args: argparse.Namespace) -> int:
    py = args.python_path or _venv_python()
    tests = [
        EXP / "tests" / "test_checkpointing.py",
        EXP / "tests" / "test_intrinsic_models.py",
        EXP / "tests" / "test_intrinsic_vec_wrapper.py",
    ]
    cmd = [py, "-m", "pytest", *[str(path) for path in tests], "-q"]
    print(f"[expr3] checkpoint preflight: {' '.join(cmd)}", flush=True)
    res = subprocess.run(cmd, cwd=str(ROOT), env=_base_env(args))
    return int(res.returncode)


def _print_plan(args: argparse.Namespace, env_steps: dict[str, int], cells: list[dict]) -> None:
    cfg = _load_minigrid_config(args)
    variants = args.variants or DEFAULT_VARIANTS
    print("[expr3] MiniGrid full matrix")
    print(f"[expr3] envs: {', '.join(env_steps)}")
    print(f"[expr3] variants: {', '.join(variants)}")
    print(f"[expr3] visible methods: {', '.join(BASELINE_METHODS + args.methods)}")
    print(f"[expr3] seeds: {', '.join(str(s) for s in args.seeds)}")
    print(f"[expr3] output: {_output_dir(args)}")
    print(f"[expr3] noise probs: {', '.join(str(p) for p in args.noise_probs)}")
    print("[expr3] budgets:")
    for env_id, steps in env_steps.items():
        print(f"  {env_id}: {steps:,} steps")
    print("[expr3] shared PPO hparams:")
    print(
        "  "
        f"policy={cfg.PPO_POLICY}, n_envs={cfg.PPO_N_ENVS}, vec_env={cfg.PPO_VEC_ENV}, "
        f"lr={cfg.PPO_HYPERPARAMS['learning_rate']}, n_steps={cfg.PPO_HYPERPARAMS['n_steps']}, "
        f"batch={cfg.PPO_HYPERPARAMS['batch_size']}, epochs={cfg.PPO_HYPERPARAMS['n_epochs']}, "
        f"gamma={cfg.PPO_HYPERPARAMS['gamma']}, gae_lambda={cfg.PPO_HYPERPARAMS['gae_lambda']}"
    )
    print("[expr3] method hparams:")
    print(f"  entropy: ent_coef={cfg.ENTROPY_COEF}")
    print(f"  RND: beta={cfg.RND_REWARD_SCALE}, lr={cfg.RND_LEARNING_RATE}, hidden={cfg.RND_HIDDEN_DIM}, out={cfg.RND_OUTPUT_DIM}")
    print(f"  LPM: beta={cfg.LPM_REWARD_SCALE}, lr={cfg.LPM_LEARNING_RATE}, hidden={cfg.LPM_HIDDEN_DIM}, buffer={cfg.LPM_BUFFER_SIZE}")
    print(
        "  "
        f"ICM: beta={cfg.ICM_REWARD_SCALE}, lr={cfg.ICM_LEARNING_RATE}, "
        f"hidden={cfg.ICM_HIDDEN_DIM}, feature={cfg.ICM_FEATURE_DIM}, "
        f"forward_loss_weight={cfg.ICM_FORWARD_LOSS_WEIGHT}"
    )
    pending = _incomplete(cells, args)
    print(f"[expr3] cells total: {len(cells)}, pending: {len(pending)}")
    if pending:
        print("[expr3] first pending cells:")
        for cell in pending[: args.preview_cells]:
            done = _progress(cell["run_name"], args)
            print(f"  {cell['run_name']} ({done:,}/{cell['steps']:,})")
        if len(pending) > args.preview_cells:
            print(f"  ... {len(pending) - args.preview_cells} more")


def main() -> int:
    ap = argparse.ArgumentParser(description="Run the MiniGrid full experiment matrix.")
    ap.add_argument("--run", action="store_true",
                    help="actually launch training; default is dry-run preview")
    ap.add_argument("--analyze-only", action="store_true",
                    help="skip training and only aggregate existing eval outputs")
    ap.add_argument("--no-analyze", action="store_true",
                    help="do not run analyze.py after completion")
    ap.add_argument("--no-preflight", action="store_true",
                    help="skip the checkpoint/resume tests before training")
    ap.add_argument("--python", dest="python_path",
                    help="Python executable for run_grid.py, train_one.py, and analyze.py")
    ap.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
                    help="experiment root containing results/ and figures/ "
                         "(default: expr_data/minigrid/exp3_final)")
    ap.add_argument("--envs", nargs="+", choices=sorted(ENV_BUDGETS),
                    help="restrict to specific MiniGrid env ids")
    ap.add_argument("--variants", nargs="+", choices=DEFAULT_VARIANTS,
                    help="restrict variants; default is clean+noisy baseline+intrinsic")
    ap.add_argument("--methods", nargs="+", default=list(INTRINSIC_METHODS),
                    choices=["rnd", "lpm", "icm", "rnd_lstm", "lpm_lstm", "icm_lstm"],
                    help="intrinsic methods; baseline variants always run none+entropy")
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--betas", type=float, nargs="+",
                    help="optional beta sweep for all intrinsic methods; default uses config defaults")
    ap.add_argument("--noise-probs", type=float, nargs="+", default=[0.10])
    ap.add_argument("--steps", type=int,
                    help="override all per-env budgets with one value")
    ap.add_argument("--doorkey-steps", type=int, default=1_000_000)
    ap.add_argument("--fourrooms-steps", type=int, default=2_000_000)
    ap.add_argument("--multiroom-steps", type=int, default=3_000_000)
    ap.add_argument("--chunk-steps", type=int, default=300_000)
    ap.add_argument("--rounds", type=int, default=0,
                    help="max chunk rounds; 0 = enough rounds for the largest requested budget")
    ap.add_argument("--jobs", type=int, default=16)
    ap.add_argument("--threads-per-job", type=int, default=1)
    ap.add_argument("--preview-cells", type=int, default=12)
    ap.add_argument("--print-run-grid-dry-run", action="store_true",
                    help="also call run_grid.py --dry-run for each selected env")
    args = ap.parse_args()

    if args.analyze_only:
        return _analyze(args)

    env_steps = _env_steps(args)
    cells = _expected_cells(args, env_steps)
    _print_plan(args, env_steps, cells)

    if not args.run:
        if args.print_run_grid_dry_run:
            for env_id, steps in env_steps.items():
                rc = _run_cmd(_runner_cmd(args, env_id, steps, dry_run=True), args)
                if rc != 0:
                    return rc
        return 0

    if not args.no_preflight:
        rc = _preflight(args)
        if rc != 0:
            print("[expr3] preflight failed; training was not started")
            return rc

    max_rounds = args.rounds or max(
        math.ceil(steps / args.chunk_steps) for steps in env_steps.values())
    for round_idx in range(1, max_rounds + 1):
        pending = _incomplete(cells, args)
        if not pending:
            print("[expr3] all cells complete")
            break

        pending_envs = sorted({cell["env"] for cell in pending})
        print(f"[expr3] round {round_idx}/{max_rounds}: {len(pending)} pending cells")
        for env_id in pending_envs:
            rc = _run_cmd(_runner_cmd(args, env_id, env_steps[env_id], dry_run=False), args)
            if rc != 0:
                return rc

    pending = _incomplete(cells, args)
    if pending:
        print(f"[expr3] stopped with {len(pending)} incomplete cells")
        print("[expr3] rerun the same command later to resume")
        return 1

    if args.no_analyze:
        return 0
    return _analyze(args)


if __name__ == "__main__":
    raise SystemExit(main())
