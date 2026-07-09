"""Validate top MiniWorld LPM hyperparameter candidates from experiment 1.

Default behavior is a dry-run preview:

    python expr2.py

Launch the validation sweep:

    python expr2.py --run

The script reads the ranked configs from an existing sweep summary, picks the
top candidates exactly, and reruns only those configs on fresh validation seeds.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parent
EXP = ROOT / "LPM_exploration" / "Miniworld" / "experiments"
RUNNER = EXP / "run_hparam_sweep.py"
SUMMARIZER = EXP / "summarize_hparam_sweep.py"
DEFAULT_SOURCE_SWEEP = ROOT / "expr_data" / "miniworld" / "sweeps" / "expr1_lpm_core_action_noise"
DEFAULT_SWEEP_NAME = "expr2_lpm_top3_action_noise_64seed"
PYTHON_EXE: str | None = None

sys.path.insert(0, str(EXP))
import run_hparam_sweep as sweep  # noqa: E402


def _venv_python() -> str:
    if PYTHON_EXE:
        return PYTHON_EXE
    linux = ROOT / "LPM_exploration" / ".venv" / "bin" / "python"
    windows = ROOT / "LPM_exploration" / ".venv" / "Scripts" / "python.exe"
    if linux.exists():
        return str(linux)
    if windows.exists():
        return str(windows)
    return sys.executable


def _as_float(row: dict[str, str], key: str, default: float) -> float:
    value = row.get(key, "")
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(out):
        return default
    return out


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _ranked_rows(source_root: Path) -> list[dict[str, str]]:
    leaderboard = source_root / "summary" / "leaderboard.csv"
    if leaderboard.exists():
        return _read_csv(leaderboard)

    summary = source_root / "summary" / "summary_by_config.csv"
    if not summary.exists():
        raise SystemExit(
            f"No leaderboard or summary_by_config found under {source_root / 'summary'}"
        )
    rows = _read_csv(summary)
    rows.sort(
        key=lambda row: (
            -_as_float(row, "coverage_frac_mean", float("-inf")),
            _as_float(row, "tv_share_mean", float("inf")),
        )
    )
    return rows


def _load_config(source_root: Path, slug: str) -> dict:
    path = source_root / "configs" / f"{slug}.json"
    if not path.exists():
        raise SystemExit(f"Missing config JSON for {slug}: {path}")
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def _select_candidates(args: argparse.Namespace) -> list[dict]:
    source_root = args.source_sweep_root.resolve()
    rows = _ranked_rows(source_root)
    by_slug = {row["config_slug"]: row for row in rows if row.get("config_slug")}

    if args.config_slugs:
        selected_rows = []
        for slug in args.config_slugs:
            if slug not in by_slug:
                raise SystemExit(f"Config slug not found in source summary: {slug}")
            selected_rows.append(by_slug[slug])
    else:
        selected_rows = rows[:args.top_k]

    selected = []
    for row in selected_rows:
        slug = row["config_slug"]
        selected.append({
            "source_slug": slug,
            "source_variant": row.get("variant") or "action_noise",
            "source_metrics": row,
            "config": _load_config(source_root, slug),
        })
    return selected


def _seeds(args: argparse.Namespace) -> list[int]:
    if args.seeds:
        return args.seeds
    return list(range(args.seed_start, args.seed_start + args.seed_count))


def _build_runs(args: argparse.Namespace, selected: list[dict], seeds: list[int]) -> tuple[Path, list[dict]]:
    root = (args.sweep_root / args.sweep_name).resolve()
    build_args = SimpleNamespace(
        steps=args.steps,
        device=args.device,
        obs_scale=args.obs_scale,
        log_interval=args.log_interval,
    )
    runs = []

    for candidate in selected:
        config = candidate["config"]
        method = config["method"]
        config_slug = sweep._slug(config)
        variants = args.variants or [candidate["source_variant"]]
        for variant in variants:
            for seed in seeds:
                rid = f"{method}-{variant}-s{seed}"
                csv_log = root / "results" / config_slug / f"{rid}.csv"
                pos_log = root / "positions" / config_slug / f"{rid}.npz"
                config_log = root / "configs" / config_slug / f"{rid}.json"
                log_path = root / "logs" / config_slug / f"{rid}.out"
                cmd = sweep._build_cmd(
                    build_args,
                    config,
                    method,
                    variant,
                    seed,
                    str(csv_log),
                    str(pos_log),
                    str(config_log),
                )
                runs.append({
                    "config": config,
                    "config_slug": config_slug,
                    "source_slug": candidate["source_slug"],
                    "run_id": rid,
                    "method": method,
                    "variant": variant,
                    "seed": seed,
                    "csv_log": str(csv_log),
                    "pos_log": str(pos_log),
                    "config_log": str(config_log),
                    "log_path": str(log_path),
                    "cmd": cmd,
                })
    return root, runs


def _write_readme(root: Path, args: argparse.Namespace, selected: list[dict], seeds: list[int]) -> None:
    path = root / "README.md"
    if path.exists():
        return
    with path.open("w", encoding="utf-8") as f:
        f.write(f"# MiniWorld validation sweep: {args.sweep_name}\n\n")
        f.write("Generated by `expr2.py`.\n\n")
        f.write(f"- Source sweep: `{args.source_sweep_root}`\n")
        f.write(f"- Selected candidates: top {len(selected)} by source leaderboard order\n")
        f.write(f"- Validation seeds: {seeds[0]}..{seeds[-1]} ({len(seeds)} seeds)\n")
        f.write(f"- Steps per run: {args.steps}\n\n")
        f.write("Selected source configs:\n\n")
        for candidate in selected:
            row = candidate["source_metrics"]
            f.write(
                f"- `{candidate['source_slug']}`: coverage={row.get('coverage_frac_mean', '')}, "
                f"tv_share={row.get('tv_share_mean', '')}\n"
            )
        f.write("\nSummarize with:\n\n")
        f.write("```bash\n")
        f.write(
            "python LPM_exploration/Miniworld/experiments/summarize_hparam_sweep.py "
            f"--sweep-root \"{root}\"\n"
        )
        f.write("```\n")


def _write_selected_csv(root: Path, selected: list[dict]) -> None:
    path = root / "selected_candidates.csv"
    metric_keys = ["coverage_frac_mean", "coverage_frac_std", "tv_share_mean", "tv_share_std"]
    fields = ["source_slug", "config_slug", "method", *metric_keys]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for candidate in selected:
            config = candidate["config"]
            row = candidate["source_metrics"]
            writer.writerow({
                "source_slug": candidate["source_slug"],
                "config_slug": sweep._slug(config),
                "method": config["method"],
                **{key: row.get(key, "") for key in metric_keys},
            })


def _write_outputs(root: Path, args: argparse.Namespace, selected: list[dict], runs: list[dict], seeds: list[int]) -> None:
    for sub in ("results", "positions", "configs", "logs"):
        (root / sub).mkdir(parents=True, exist_ok=True)
    configs = {sweep._slug(candidate["config"]): candidate["config"] for candidate in selected}
    sweep._write_configs(str(root), configs)
    sweep._write_manifest(str(root), runs)
    _write_readme(root, args, selected, seeds)
    _write_selected_csv(root, selected)


def _base_env() -> dict[str, str]:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(EXP) + os.pathsep + env.get("PYTHONPATH", "")
    env.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    env.setdefault("PYGLET_HEADLESS", "true")
    return env


def _summarize(root: Path) -> int:
    cmd = [_venv_python(), str(SUMMARIZER), "--sweep-root", str(root)]
    print(f"[expr2] summarize: {' '.join(cmd)}", flush=True)
    res = subprocess.run(cmd, cwd=str(ROOT), env=_base_env())
    return int(res.returncode)


def _print_preview(root: Path, selected: list[dict], runs: list[dict], pending: list[dict], args: argparse.Namespace) -> None:
    print(f"[expr2] source: {args.source_sweep_root}")
    print(f"[expr2] output: {root}")
    print("[expr2] selected candidates:")
    for candidate in selected:
        row = candidate["source_metrics"]
        config = candidate["config"]
        print(
            "  "
            f"{candidate['source_slug']} -> {sweep._slug(config)} "
            f"(method={config['method']}, coverage={row.get('coverage_frac_mean', '')}, "
            f"tv_share={row.get('tv_share_mean', '')})"
        )
    print(f"[expr2] planned runs: {len(runs)}, pending: {len(pending)}")
    sample = pending if args.print_all else pending[: args.preview_runs]
    for run in sample:
        print(f"[run] {run['config_slug']} / {run['run_id']}")
        print("      " + " ".join(run["cmd"]))
    if len(sample) < len(pending):
        print(f"[expr2] ... {len(pending) - len(sample)} more pending runs hidden")


def main() -> int:
    ap = argparse.ArgumentParser(description="Validate top MiniWorld hyperparameter candidates.")
    ap.add_argument("--run", action="store_true",
                    help="actually launch runs; default is dry-run preview")
    ap.add_argument("--summarize-only", action="store_true",
                    help="skip training and summarize the validation sweep folder")
    ap.add_argument("--source-sweep-root", type=Path, default=DEFAULT_SOURCE_SWEEP,
                    help="sweep root containing summary/leaderboard.csv and configs/*.json")
    ap.add_argument("--sweep-root", type=Path,
                    default=ROOT / "expr_data" / "miniworld" / "sweeps")
    ap.add_argument("--sweep-name", default=DEFAULT_SWEEP_NAME)
    ap.add_argument("--top-k", type=int, default=3,
                    help="number of leaderboard candidates to validate")
    ap.add_argument("--config-slugs", nargs="+",
                    help="validate these exact source config slugs instead of --top-k")
    ap.add_argument("--variants", nargs="+", choices=sweep.VARIANTS,
                    help="override variants; default reuses each source row's variant")
    ap.add_argument("--seeds", type=int, nargs="+",
                    help="explicit validation seeds")
    ap.add_argument("--seed-start", type=int, default=9,
                    help="first validation seed when --seeds is omitted")
    ap.add_argument("--seed-count", type=int, default=64,
                    help="number of validation seeds when --seeds is omitted")
    ap.add_argument("--steps", type=int, default=50000)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--python", dest="python_path",
                    help="Python executable for train_maze.py and summarization")
    ap.add_argument("--obs-scale", type=float, default=1.0)
    ap.add_argument("--log-interval", type=int, default=5)
    ap.add_argument("--jobs", type=int, default=96)
    ap.add_argument("--threads-per-job", type=int, default=1)
    ap.add_argument("--max-runs", type=int, default=0,
                    help="debug cap after skip filtering; 0 = no cap")
    ap.add_argument("--force", action="store_true",
                    help="rerun even when a position log already exists")
    ap.add_argument("--no-summarize", action="store_true",
                    help="do not summarize automatically after a successful --run")
    ap.add_argument("--preview-runs", type=int, default=8,
                    help="number of pending commands to print in dry-run mode")
    ap.add_argument("--print-all", action="store_true",
                    help="print every pending command during dry-run")
    args = ap.parse_args()

    global PYTHON_EXE
    PYTHON_EXE = args.python_path
    sweep.PY = _venv_python()

    if args.summarize_only:
        root = (args.sweep_root / args.sweep_name).resolve()
        return _summarize(root)

    selected = _select_candidates(args)
    if not selected:
        raise SystemExit("No candidates selected")
    seeds = _seeds(args)
    if not seeds:
        raise SystemExit("No validation seeds selected")
    root, runs = _build_runs(args, selected, seeds)

    pending = [run for run in runs if args.force or not Path(run["pos_log"]).exists()]
    if args.max_runs > 0:
        pending = pending[: args.max_runs]

    _print_preview(root, selected, runs, pending, args)
    if not args.run:
        return 0

    _write_outputs(root, args, selected, runs, seeds)
    if not pending:
        print("[expr2] no pending runs")
        return 0 if args.no_summarize else _summarize(root)

    threads = args.threads_per_job
    base_env = _base_env()
    failures = 0
    if args.jobs <= 1:
        for run in pending:
            print(f"[run] {run['config_slug']} / {run['run_id']}", flush=True)
            _, _, rc = sweep._run_one(run, base_env, threads)
            if rc != 0:
                failures += 1
                print(f"[FAIL] {run['config_slug']} / {run['run_id']} (see {run['log_path']})")
    else:
        with ThreadPoolExecutor(max_workers=args.jobs) as ex:
            futs = {ex.submit(sweep._run_one, run, base_env, threads): run for run in pending}
            for fut in as_completed(futs):
                slug, rid, rc = fut.result()
                tag = "done" if rc == 0 else "FAIL"
                if rc != 0:
                    failures += 1
                log_path = futs[fut]["log_path"]
                print(f"[{tag}] {slug} / {rid}" + ("" if rc == 0 else f" (see {log_path})"))

    if failures:
        print(f"[expr2] {failures} runs failed; skipping automatic summary")
        return 1
    if args.no_summarize:
        return 0
    return _summarize(root)


if __name__ == "__main__":
    raise SystemExit(main())
