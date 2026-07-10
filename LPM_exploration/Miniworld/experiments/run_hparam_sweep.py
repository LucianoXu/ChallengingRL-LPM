"""Hyperparameter sweep runner for the MiniWorld maze experiment.

This script keeps the paper-faithful `run_grid.py` intact and adds a separate
diagnostic runner for "why did this reproduction behave this way?" questions.
It writes each hyperparameter configuration into its own directory under
<repo>/expr_data/miniworld/sweeps/<sweep-name>/, so ordinary analysis can still
be run on any single configuration without filename collisions.

Examples:
  python run_hparam_sweep.py --preset smoke --dry-run

  python run_hparam_sweep.py --preset lpm_core

After a sweep, summarize with:
  python summarize_hparam_sweep.py --sweep-root ../../../expr_data/miniworld/sweeps/lpm_lambda_entropy
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed


EXP = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(EXP)))
DEFAULT_SWEEP_ROOT = os.path.join(REPO_ROOT, "expr_data", "miniworld", "sweeps")

METHODS = ["lpm", "rnd", "icm", "mse", "none", "uniform"]
VARIANTS = ["nonoise", "noisy_tv", "action_noise"]

HARD_CODED_SWEEPS = {
    "smoke": {
        "sweep_name": "lpm_smoke",
        "steps": 512,
        "seeds": [1],
        "methods": ["lpm"],
        "variants": ["action_noise"],
        "device": "cpu",
        "obs_scale": 1.0,
        "log_interval": 1,
        "jobs": 1,
        "threads_per_job": 0,
        "max_runs": 0,
        "lambda_intrinsic_values": [0.3, 1.0],
        "entropy_coef_values": [0.03],
        "normalize_intrinsic_values": ["off"],
        "policy_lr_values": [0.01],
        "gamma_values": [0.99],
        "gae_lambda_values": [0.95],
        "value_loss_coef_values": [0.5],
        "max_grad_norm_values": [0.5],
        "update_frequency_values": [64],
        "lpm_reward_space_values": ["log"],
        "lpm_buffer_size_values": [100],
        "lpm_update_unc_every_values": ["auto"],
        "lpm_eta_values": [1.0],
        "lpm_pred_lr_values": [1e-3],
        "lpm_unc_lr_values": ["auto"],
        "mse_lr_values": [1e-3],
        "rnd_lr_values": [1e-3],
        "rnd_emb_values": [256],
        "icm_lr_values": [1e-3],
        "icm_beta_values": [0.2],
    },
    "lpm_core": {
        "sweep_name": "lpm_core_action_noise",
        "steps": 50000,
        "seeds": [1, 2, 3, 4, 5, 6, 7, 8],
        "methods": ["lpm"],
        "variants": ["action_noise"],
        "device": "cpu",
        "obs_scale": 1.0,
        "log_interval": 5,
        "jobs": 96,
        "threads_per_job": 1,
        "max_runs": 0,
        "lambda_intrinsic_values": [0.1, 0.3, 1.0, 3.0],
        "entropy_coef_values": [0.01, 0.03, 0.05],
        "normalize_intrinsic_values": ["off"],
        "policy_lr_values": [0.01],
        "gamma_values": [0.99],
        "gae_lambda_values": [0.95],
        "value_loss_coef_values": [0.5],
        "max_grad_norm_values": [0.5],
        "update_frequency_values": [64],
        "lpm_reward_space_values": ["log"],
        "lpm_buffer_size_values": [100],
        "lpm_update_unc_every_values": ["auto"],
        "lpm_eta_values": [1.0],
        "lpm_pred_lr_values": [1e-3],
        "lpm_unc_lr_values": ["auto"],
        "mse_lr_values": [1e-3],
        "rnd_lr_values": [1e-3],
        "rnd_emb_values": [256],
        "icm_lr_values": [1e-3],
        "icm_beta_values": [0.2],
    },
}


def _product_dict(grid):
    keys = list(grid)
    for vals in itertools.product(*(grid[k] for k in keys)):
        yield dict(zip(keys, vals))


def _apply_preset(args):
    preset = HARD_CODED_SWEEPS[args.preset]
    for key, value in preset.items():
        if getattr(args, key) is None:
            if isinstance(value, list):
                value = list(value)
            setattr(args, key, value)
    if args.sweep_root is None:
        args.sweep_root = DEFAULT_SWEEP_ROOT


def _parse_auto_ints(values):
    parsed = []
    for v in values:
        if str(v).lower() == "auto":
            parsed.append(None)
        else:
            iv = int(v)
            if iv <= 0:
                raise argparse.ArgumentTypeError("update cadence values must be positive or 'auto'")
            parsed.append(iv)
    return parsed


def _parse_auto_floats(values):
    parsed = []
    for v in values:
        if str(v).lower() == "auto":
            parsed.append(None)
        else:
            fv = float(v)
            if fv <= 0:
                raise argparse.ArgumentTypeError("learning-rate values must be positive or 'auto'")
            parsed.append(fv)
    return parsed


def _fmt_value(value):
    if value is None:
        return "auto"
    if isinstance(value, bool):
        return "on" if value else "off"
    if isinstance(value, float):
        text = f"{value:.6g}"
        return text.replace("-", "m").replace(".", "p")
    return str(value).replace("-", "m").replace(".", "p").replace("/", "_")


def _slug(config):
    raw = json.dumps(config, sort_keys=True)
    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:10]
    method = config["method"]
    parts = [
        f"m-{_fmt_value(method)}",
        f"lam-{_fmt_value(config['lambda_intrinsic'])}",
        f"ent-{_fmt_value(config['entropy_coef'])}",
    ]
    if config.get("normalize_intrinsic"):
        parts.append("norm-on")
    if config.get("update_frequency") != 64:
        parts.append(f"upd-{_fmt_value(config['update_frequency'])}")
    if method == "lpm":
        parts.extend([
            f"space-{_fmt_value(config['lpm_reward_space'])}",
            f"buf-{_fmt_value(config['lpm_buffer_size'])}",
        ])
        if config.get("lpm_unc_lr") is not None:
            parts.append(f"unclr-{_fmt_value(config['lpm_unc_lr'])}")
    elif method == "rnd" and config.get("rnd_lr") != 1e-3:
        parts.append(f"rndlr-{_fmt_value(config['rnd_lr'])}")
    elif method == "icm" and config.get("icm_lr") != 1e-3:
        parts.append(f"icmlr-{_fmt_value(config['icm_lr'])}")
    elif method == "mse" and config.get("mse_lr") != 1e-3:
        parts.append(f"mselr-{_fmt_value(config['mse_lr'])}")
    parts.append(f"h-{digest}")
    return "__".join(parts)


def _common_configs(args):
    return _product_dict({
        "lambda_intrinsic": args.lambda_intrinsic_values,
        "entropy_coef": args.entropy_coef_values,
        "normalize_intrinsic": [v == "on" for v in args.normalize_intrinsic_values],
        "policy_lr": args.policy_lr_values,
        "gamma": args.gamma_values,
        "gae_lambda": args.gae_lambda_values,
        "value_loss_coef": args.value_loss_coef_values,
        "max_grad_norm": args.max_grad_norm_values,
        "update_frequency": args.update_frequency_values,
    })


def _model_configs(method, args):
    if method == "lpm":
        return _product_dict({
            "lpm_reward_space": args.lpm_reward_space_values,
            "lpm_buffer_size": args.lpm_buffer_size_values,
            "lpm_update_unc_every": _parse_auto_ints(args.lpm_update_unc_every_values),
            "lpm_eta": args.lpm_eta_values,
            "lpm_pred_lr": args.lpm_pred_lr_values,
            "lpm_unc_lr": _parse_auto_floats(args.lpm_unc_lr_values),
        })
    if method == "mse":
        return _product_dict({"mse_lr": args.mse_lr_values})
    if method == "rnd":
        return _product_dict({"rnd_lr": args.rnd_lr_values, "rnd_emb": args.rnd_emb_values})
    if method == "icm":
        return _product_dict({"icm_lr": args.icm_lr_values, "icm_beta": args.icm_beta_values})
    return iter([{}])


def _add(cmd, name, value):
    cmd.extend([name, str(value)])


def _build_cmd(args, config, method, variant, seed, csv_log, pos_log, config_log):
    train_method = "none" if method == "uniform" else method
    cmd = [
        PY,
        os.path.join(EXP, "train_maze.py"),
        "--method", train_method,
        "--variant", variant,
        "--seed", str(seed),
        "--steps", str(args.steps),
        "--device", args.device,
        "--obs-scale", str(args.obs_scale),
        "--csv-log", csv_log,
        "--pos-log", pos_log,
        "--config-log", config_log,
        "--log-interval", str(args.log_interval),
    ]
    _add(cmd, "--update-frequency", config["update_frequency"])
    _add(cmd, "--lambda-intrinsic", config["lambda_intrinsic"])
    _add(cmd, "--entropy-coef", config["entropy_coef"])
    _add(cmd, "--policy-lr", config["policy_lr"])
    _add(cmd, "--gamma", config["gamma"])
    _add(cmd, "--gae-lambda", config["gae_lambda"])
    _add(cmd, "--value-loss-coef", config["value_loss_coef"])
    _add(cmd, "--max-grad-norm", config["max_grad_norm"])
    if config["normalize_intrinsic"]:
        cmd.append("--normalize-intrinsic")

    if method == "uniform":
        cmd.append("--random-policy")
    elif method == "lpm":
        _add(cmd, "--lpm-reward-space", config["lpm_reward_space"])
        _add(cmd, "--lpm-buffer-size", config["lpm_buffer_size"])
        if config["lpm_update_unc_every"] is not None:
            _add(cmd, "--lpm-update-unc-every", config["lpm_update_unc_every"])
        _add(cmd, "--lpm-eta", config["lpm_eta"])
        _add(cmd, "--lpm-pred-lr", config["lpm_pred_lr"])
        if config["lpm_unc_lr"] is not None:
            _add(cmd, "--lpm-unc-lr", config["lpm_unc_lr"])
    elif method == "mse":
        _add(cmd, "--mse-lr", config["mse_lr"])
    elif method == "rnd":
        _add(cmd, "--rnd-lr", config["rnd_lr"])
        _add(cmd, "--rnd-emb", config["rnd_emb"])
    elif method == "icm":
        _add(cmd, "--icm-lr", config["icm_lr"])
        _add(cmd, "--icm-beta", config["icm_beta"])
    return cmd


def _is_complete(path):
    return os.path.exists(path)


def _write_readme(root, args):
    path = os.path.join(root, "README.md")
    if os.path.exists(path):
        return
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"# MiniWorld hyperparameter sweep: {args.sweep_name}\n\n")
        f.write("Generated by `LPM_exploration/Miniworld/experiments/run_hparam_sweep.py`.\n\n")
        f.write("Layout:\n\n")
        f.write("- `results/<config_slug>/*.csv`: per-update training metrics.\n")
        f.write("- `positions/<config_slug>/*.npz`: per-step positions/actions.\n")
        f.write("- `configs/<config_slug>.json`: hyperparameters for that config.\n")
        f.write("- `configs/<config_slug>/<run_id>.json`: resolved run config written by `train_maze.py`.\n")
        f.write("- `logs/<config_slug>/<run_id>.out`: stdout/stderr for each run.\n")
        f.write("- `manifest.csv`: all planned runs for this sweep launch.\n\n")
        f.write("Summarize with:\n\n")
        f.write("```bash\n")
        f.write("python LPM_exploration/Miniworld/experiments/summarize_hparam_sweep.py ")
        f.write(f"--sweep-root \"{root}\"\n")
        f.write("```\n")


def _write_manifest(root, rows):
    path = os.path.join(root, "manifest.csv")
    fields = ["config_slug", "method", "variant", "seed", "csv_log", "pos_log", "log_path"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in rows:
            w.writerow({k: row[k] for k in fields})


def _write_configs(root, configs):
    config_dir = os.path.join(root, "configs")
    os.makedirs(config_dir, exist_ok=True)
    for slug, config in sorted(configs.items()):
        with open(os.path.join(config_dir, slug + ".json"), "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, sort_keys=True)


def _run_one(run, base_env, threads):
    env = dict(base_env)
    for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
              "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
              # local fix: pin Mesa llvmpipe's software-GL render pool too, else
              # each headless job spawns ~1 thread/core (~32) and N jobs oversubscribe
              # the box (load >200, ~4x slower). See UPSTREAM.md 2026-07-09.
              "LP_NUM_THREADS"):
        env[k] = str(threads)
    os.makedirs(os.path.dirname(run["log_path"]), exist_ok=True)
    with open(run["log_path"], "w", encoding="utf-8") as out:
        res = subprocess.run(run["cmd"], env=env, cwd=EXP, stdout=out, stderr=subprocess.STDOUT)
    return run["config_slug"], run["run_id"], res.returncode


def _build_runs(args):
    root = os.path.abspath(os.path.join(args.sweep_root, args.sweep_name))
    runs = []
    configs = {}

    for method in args.methods:
        for common in _common_configs(args):
            for model_cfg in _model_configs(method, args):
                config = {"method": method, **common, **model_cfg}
                config_slug = _slug(config)
                configs[config_slug] = config
                for variant, seed in itertools.product(args.variants, args.seeds):
                    rid = f"{method}-{variant}-s{seed}"
                    csv_log = os.path.join(root, "results", config_slug, rid + ".csv")
                    pos_log = os.path.join(root, "positions", config_slug, rid + ".npz")
                    config_log = os.path.join(root, "configs", config_slug, rid + ".json")
                    log_path = os.path.join(root, "logs", config_slug, rid + ".out")
                    cmd = _build_cmd(args, config, method, variant, seed, csv_log, pos_log, config_log)
                    runs.append({
                        "config": config,
                        "config_slug": config_slug,
                        "run_id": rid,
                        "method": method,
                        "variant": variant,
                        "seed": seed,
                        "csv_log": csv_log,
                        "pos_log": pos_log,
                        "config_log": config_log,
                        "log_path": log_path,
                        "cmd": cmd,
                    })

    return root, runs, configs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", choices=sorted(HARD_CODED_SWEEPS), default="smoke",
                    help="hardcoded sweep recipe to use; individual CLI flags still override it")
    ap.add_argument("--sweep-name")
    ap.add_argument("--sweep-root")
    ap.add_argument("--steps", type=int)
    ap.add_argument("--seeds", type=int, nargs="+")
    ap.add_argument("--methods", nargs="+", choices=METHODS)
    ap.add_argument("--variants", nargs="+", choices=VARIANTS)
    ap.add_argument("--device")
    ap.add_argument("--obs-scale", type=float)
    ap.add_argument("--log-interval", type=int)
    ap.add_argument("--jobs", type=int)
    ap.add_argument("--threads-per-job", type=int,
                    help="CPU threads per worker; 0 = auto (cores // jobs, min 1)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force", action="store_true",
                    help="rerun even when the final position log already exists")
    ap.add_argument("--max-runs", type=int,
                    help="debug cap after skip filtering; 0 = no cap")

    ap.add_argument("--lambda-intrinsic-values", type=float, nargs="+")
    ap.add_argument("--entropy-coef-values", type=float, nargs="+")
    ap.add_argument("--normalize-intrinsic-values", choices=["off", "on"], nargs="+")
    ap.add_argument("--policy-lr-values", type=float, nargs="+")
    ap.add_argument("--gamma-values", type=float, nargs="+")
    ap.add_argument("--gae-lambda-values", type=float, nargs="+")
    ap.add_argument("--value-loss-coef-values", type=float, nargs="+")
    ap.add_argument("--max-grad-norm-values", type=float, nargs="+")
    ap.add_argument("--update-frequency-values", type=int, nargs="+")

    ap.add_argument("--lpm-reward-space-values", choices=["log", "raw"], nargs="+")
    ap.add_argument("--lpm-buffer-size-values", type=int, nargs="+")
    ap.add_argument("--lpm-update-unc-every-values", nargs="+")
    ap.add_argument("--lpm-eta-values", type=float, nargs="+")
    ap.add_argument("--lpm-pred-lr-values", type=float, nargs="+")
    ap.add_argument("--lpm-unc-lr-values", nargs="+")
    ap.add_argument("--mse-lr-values", type=float, nargs="+")
    ap.add_argument("--rnd-lr-values", type=float, nargs="+")
    ap.add_argument("--rnd-emb-values", type=int, nargs="+")
    ap.add_argument("--icm-lr-values", type=float, nargs="+")
    ap.add_argument("--icm-beta-values", type=float, nargs="+")
    args = ap.parse_args()
    _apply_preset(args)

    root, runs, configs = _build_runs(args)

    pending = [r for r in runs if args.force or not _is_complete(r["pos_log"])]
    if args.max_runs > 0:
        pending = pending[:args.max_runs]

    cores = os.cpu_count() or 4
    threads = args.threads_per_job or max(1, cores // max(1, args.jobs))
    print(f"{len(runs)} planned runs, {len(pending)} to run, "
          f"jobs={args.jobs}, threads/job={threads}, root={root}")

    if args.dry_run:
        for run in pending:
            print(f"[run] {run['config_slug']} / {run['run_id']}")
            print("      " + " ".join(run["cmd"]))
        return

    for sub in ("results", "positions", "configs", "logs"):
        os.makedirs(os.path.join(root, sub), exist_ok=True)
    _write_readme(root, args)
    _write_configs(root, configs)
    _write_manifest(root, runs)

    base_env = dict(os.environ)
    base_env["PYTHONPATH"] = EXP + os.pathsep + base_env.get("PYTHONPATH", "")
    base_env.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    base_env.setdefault("PYGLET_HEADLESS", "true")

    if args.jobs <= 1:
        for run in pending:
            print(f"[run] {run['config_slug']} / {run['run_id']}")
            _, _, rc = _run_one(run, base_env, threads)
            if rc != 0:
                print(f"[FAIL] {run['config_slug']} / {run['run_id']} (see {run['log_path']})")
        return

    with ThreadPoolExecutor(max_workers=args.jobs) as ex:
        futs = {ex.submit(_run_one, run, base_env, threads): run for run in pending}
        for fut in as_completed(futs):
            slug, rid, rc = fut.result()
            tag = "done" if rc == 0 else "FAIL"
            log_path = futs[fut]["log_path"]
            print(f"[{tag}] {slug} / {rid}" + ("" if rc == 0 else f" (see {log_path})"))


if __name__ == "__main__":
    main()
