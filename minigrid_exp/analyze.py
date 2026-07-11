"""Aggregate MiniGrid runs into reusable result tables.

Reads SB3 EvalCallback outputs under
expr_data/minigrid/results/logs/<algo>/eval/<run>/<chunk>/
and writes summary CSVs under expr_data/minigrid/figures/. Legacy diagnostic
plots are available behind ``--diagnostics``; report-ready figures are produced
by ``reports/make_publication_figures.py``.

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

METHOD_ORDER = ["none", "entropy", "rnd", "icm", "lpm", "rnd_lstm", "icm_lstm", "lpm_lstm"]
ENV_ORDER = [
    "MiniGrid-DoorKey-5x5-v0",
    "MiniGrid-FourRooms-v0",
    "MiniGrid-MultiRoom-N6-v0",
]

RUN_RE = re.compile(
    r"^(?P<env>.+?)__(?P<variant>baseline_no_noise|baseline_noise|"
    r"intrinsic_no_noise|intrinsic_noise)__"
    r"(?P<method>rnd_lstm|lpm_lstm|icm_lstm|rnd|lpm|icm|count|entropy|none)__seed_(?P<seed>\d+)"
    r"(?:__beta(?P<beta>[0-9.eE+-]+))?(?:__np(?P<np>[0-9.eE+-]+))?$")


def parse_run_name(name: str):
    m = RUN_RE.match(name)
    if not m:
        return None
    d = m.groupdict()
    return {"env": d["env"], "variant": d["variant"], "method": d["method"],
            "seed": int(d["seed"]), "beta": d["beta"], "np": d["np"]}


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
    g = df.groupby(["env", "variant", "method", "beta", "np", "timestep"], dropna=False)
    return g["mean_return"].agg(["mean", "std", "count"]).reset_index()


def plot_curves(summary: pd.DataFrame, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    for env in sorted(summary["env"].unique()):
        sub = summary[summary["env"] == env]
        plt.figure(figsize=(7, 5))
        for (variant, method, beta, npv), s in sub.groupby(["variant", "method", "beta", "np"], dropna=False):
            s = s.sort_values("timestep")
            label = f"{variant}/{method}" + (f"/b{beta}" if beta else "") + (f"/np{npv}" if npv else "")
            plt.plot(s["timestep"], s["mean"], label=label)
            plt.fill_between(s["timestep"], s["mean"] - s["std"], s["mean"] + s["std"], alpha=0.15)
        plt.xlabel("training step"); plt.ylabel("eval mean return")
        plt.title(env); plt.legend(fontsize=6); plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"fig_sample_efficiency_{env.replace('/', '_')}.png"), dpi=130)
        plt.close()


def final_success_table(df: pd.DataFrame, frac: float = 0.1) -> pd.DataFrame:
    fdf = per_seed_final_returns(df, frac=frac)
    return (fdf.groupby(["env", "variant", "method", "beta", "np"], dropna=False)["final_return"]
            .agg(["mean", "std", "count"]).reset_index())


def per_seed_final_returns(df: pd.DataFrame, frac: float = 0.1) -> pd.DataFrame:
    out = []
    for keys, s in df.groupby(["env", "variant", "method", "beta", "np", "seed"], dropna=False):
        s = s.sort_values("timestep")
        k = max(1, int(len(s) * frac))
        out.append({**dict(zip(["env", "variant", "method", "beta", "np", "seed"], keys)),
                    "final_return": s["mean_return"].tail(k).mean()})
    return pd.DataFrame(out)


def matrix_stats_table(per_seed: pd.DataFrame) -> pd.DataFrame:
    """Aggregate the experiment matrix with variance and solve-rate columns."""
    rows = []
    group_cols = ["env", "variant", "method", "beta", "np"]
    for keys, s in per_seed.groupby(group_cols, dropna=False):
        vals = s["final_return"].to_numpy(dtype=float)
        rows.append({
            **dict(zip(group_cols, keys)),
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
            "sem": float(np.std(vals, ddof=1) / np.sqrt(len(vals))) if len(vals) > 1 else 0.0,
            "median": float(np.median(vals)),
            "iqr": float(np.percentile(vals, 75) - np.percentile(vals, 25)),
            "count": int(len(vals)),
            "solve_rate_0p5": float(np.mean(vals >= 0.5)),
            "zero_rate_0p05": float(np.mean(vals <= 0.05)),
        })
    return pd.DataFrame(rows)


def plot_matrix_heatmaps(matrix: pd.DataFrame, out_dir: str):
    """Plot env x method heatmaps for default-beta matrix cells."""
    os.makedirs(out_dir, exist_ok=True)
    default_beta = matrix[matrix["beta"].isna()]
    if default_beta.empty:
        return
    for (variant, npv), sub in default_beta.groupby(["variant", "np"], dropna=False):
        methods = [m for m in METHOD_ORDER if m in set(sub["method"])]
        envs = [e for e in ENV_ORDER if e in set(sub["env"])]
        if not methods or not envs:
            continue
        pivot = sub.pivot_table(index="env", columns="method", values="mean", aggfunc="mean")
        arr = pivot.reindex(index=envs, columns=methods).to_numpy(dtype=float)
        fig, ax = plt.subplots(figsize=(1.25 * len(methods) + 2.5, 0.55 * len(envs) + 2.0))
        im = ax.imshow(arr, vmin=0.0, vmax=1.0, cmap="viridis")
        ax.set_xticks(np.arange(len(methods)))
        ax.set_xticklabels(methods, rotation=30, ha="right")
        ax.set_yticks(np.arange(len(envs)))
        ax.set_yticklabels([e.replace("MiniGrid-", "").replace("-v0", "") for e in envs])
        title_np = "clean" if pd.isna(npv) else f"noise p={npv}"
        ax.set_title(f"Final return matrix: {variant}, {title_np}")
        for i in range(arr.shape[0]):
            for j in range(arr.shape[1]):
                if np.isfinite(arr[i, j]):
                    ax.text(j, i, f"{arr[i, j]:.2f}", ha="center", va="center",
                            color="white" if arr[i, j] < 0.55 else "black", fontsize=8)
        fig.colorbar(im, ax=ax, label="mean final eval return")
        fig.tight_layout()
        safe_np = "clean" if pd.isna(npv) else f"np{str(npv).replace('.', 'p')}"
        out = os.path.join(out_dir, f"fig_matrix_{variant}_{safe_np}.png")
        fig.savefig(out, dpi=140)
        plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", default=str(config.LOGS_DIR))
    ap.add_argument("--figures", default=str(config.EXPR_DATA / "figures"))
    ap.add_argument(
        "--diagnostics",
        action="store_true",
        help="also write the legacy all-config curves and matrix heatmaps",
    )
    a = ap.parse_args()
    df = aggregate_eval_curves(a.logs)
    if df.empty:
        print("no eval data found under", a.logs); return
    print(f"loaded {len(df)} eval rows")
    summary = summarize(df)
    os.makedirs(a.figures, exist_ok=True)
    per_seed = per_seed_final_returns(df)
    per_seed.to_csv(os.path.join(a.figures, "table_final_by_seed.csv"), index=False)
    final = final_success_table(df)
    final.to_csv(os.path.join(a.figures, "table_final_success.csv"), index=False)
    matrix = matrix_stats_table(per_seed)
    matrix.to_csv(os.path.join(a.figures, "table_matrix_stats.csv"), index=False)
    if a.diagnostics:
        plot_curves(summary, a.figures)
        plot_matrix_heatmaps(matrix, a.figures)
    print("wrote final/matrix tables to", a.figures)
    if not a.diagnostics:
        print("publication figures: python reports/make_publication_figures.py --domain minigrid")


if __name__ == "__main__":
    main()
