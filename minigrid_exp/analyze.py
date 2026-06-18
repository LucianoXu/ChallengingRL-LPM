"""Aggregate MiniGrid runs -> sample-efficiency curves + final-success table.

Reads SB3 EvalCallback outputs under expr_data/minigrid/logs/<algo>/eval/<run>/<chunk>/
and writes figures + a summary CSV under expr_data/minigrid/figures/.

Each run may have one or more chunk dirs named c0, c300000, c600000, ... The
per-chunk timesteps are already global (reset_num_timesteps=False after c0), so
concatenating them gives the full eval curve.
"""
from __future__ import annotations

import argparse
import glob
import os
import re
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config

RUN_RE = re.compile(
    r"^(?P<env>.+?)__(?P<variant>baseline_no_noise|baseline_noise|"
    r"intrinsic_no_noise|intrinsic_noise)__(?P<method>rnd|lpm|count|entropy|none)__seed_(?P<seed>\d+)"
    r"(?:__beta(?P<beta>[0-9.eE+-]+))?$")


def parse_run_name(name: str):
    m = RUN_RE.match(name)
    if not m:
        return None
    d = m.groupdict()
    return {"env": d["env"], "variant": d["variant"], "method": d["method"],
            "seed": int(d["seed"]), "beta": d["beta"]}


def load_eval_npz(path: str):
    data = np.load(path)
    ts = data["timesteps"]
    res = data["results"]  # (n_evals, n_episodes)
    return [{"timestep": int(t), "mean_return": float(r.mean())}
            for t, r in zip(ts, res)]


def aggregate_eval_curves(logs_dir: str) -> pd.DataFrame:
    """Glob eval/<run_name>/<chunk>/evaluations.npz, concatenate chunks per run."""
    rows = []

    # Collect all chunk npz paths grouped by run_name
    # Pattern: eval/<run_name>/<chunk_dir>/evaluations.npz
    npz_pattern = os.path.join(logs_dir, "eval", "*", "*", "evaluations.npz")
    chunks_by_run: dict[str, list[str]] = defaultdict(list)
    for npz in glob.glob(npz_pattern):
        # dirname = <logs_dir>/eval/<run_name>/<chunk_dir>
        chunk_dir = os.path.dirname(npz)
        run_dir = os.path.dirname(chunk_dir)
        run_name = os.path.basename(run_dir)
        chunks_by_run[run_name].append(npz)

    # Also handle legacy single-level eval/<run_name>/evaluations.npz
    legacy_pattern = os.path.join(logs_dir, "eval", "*", "evaluations.npz")
    for npz in glob.glob(legacy_pattern):
        run_name = os.path.basename(os.path.dirname(npz))
        # Only treat as legacy if not already seen via chunked path
        if run_name not in chunks_by_run:
            chunks_by_run[run_name].append(npz)

    for run_name, npz_paths in chunks_by_run.items():
        meta = parse_run_name(run_name)
        if meta is None:
            continue
        # Sort chunks by the numeric offset in the chunk dir name (c0, c300000, ...)
        # For legacy single-level, there is only one path.
        def chunk_order(p):
            chunk_dir_name = os.path.basename(os.path.dirname(p))
            if chunk_dir_name.startswith("c") and chunk_dir_name[1:].isdigit():
                return int(chunk_dir_name[1:])
            return -1  # legacy: sort before any chunk

        npz_paths_sorted = sorted(npz_paths, key=chunk_order)
        # Deduplicate timesteps across chunks (keep first occurrence)
        seen_ts: set[int] = set()
        for npz in npz_paths_sorted:
            for pt in load_eval_npz(npz):
                if pt["timestep"] not in seen_ts:
                    seen_ts.add(pt["timestep"])
                    rows.append({**meta, **pt})

    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby(["env", "variant", "method", "beta", "timestep"], dropna=False)
    return g["mean_return"].agg(["mean", "std", "count"]).reset_index()


def plot_curves(summary: pd.DataFrame, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    for env in sorted(summary["env"].unique()):
        sub = summary[summary["env"] == env]
        plt.figure(figsize=(7, 5))
        for (variant, method, beta), s in sub.groupby(["variant", "method", "beta"], dropna=False):
            s = s.sort_values("timestep")
            label = f"{variant}/{method}" + (f"/b{beta}" if beta else "")
            plt.plot(s["timestep"], s["mean"], label=label)
            plt.fill_between(s["timestep"], s["mean"] - s["std"], s["mean"] + s["std"], alpha=0.15)
        plt.xlabel("training step"); plt.ylabel("eval mean return")
        plt.title(env); plt.legend(fontsize=6); plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"fig_sample_efficiency_{env.replace('/', '_')}.png"), dpi=130)
        plt.close()


def final_success_table(df: pd.DataFrame, frac: float = 0.1) -> pd.DataFrame:
    out = []
    for keys, s in df.groupby(["env", "variant", "method", "beta", "seed"], dropna=False):
        s = s.sort_values("timestep")
        k = max(1, int(len(s) * frac))
        out.append({**dict(zip(["env", "variant", "method", "beta", "seed"], keys)),
                    "final_return": s["mean_return"].tail(k).mean()})
    fdf = pd.DataFrame(out)
    return (fdf.groupby(["env", "variant", "method", "beta"], dropna=False)["final_return"]
            .agg(["mean", "std", "count"]).reset_index())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", default=str(config.LOGS_DIR))
    ap.add_argument("--figures", default=str(config.EXPR_DATA / "figures"))
    a = ap.parse_args()
    df = aggregate_eval_curves(a.logs)
    if df.empty:
        print("no eval data found under", a.logs); return
    print(f"loaded {len(df)} eval rows")
    summary = summarize(df)
    plot_curves(summary, a.figures)
    os.makedirs(a.figures, exist_ok=True)
    final_success_table(df).to_csv(os.path.join(a.figures, "table_final_success.csv"), index=False)
    print("wrote figures + table_final_success.csv to", a.figures)


if __name__ == "__main__":
    main()
