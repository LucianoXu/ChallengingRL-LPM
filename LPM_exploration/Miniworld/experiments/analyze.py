"""Aggregate maze runs into a coverage table + curves + heatmap-evolution figures.

Usage: python analyze.py  (uses ./results, ./positions -> ./figures)
"""
from __future__ import annotations

import argparse
import glob
import os
import re

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import heatmaps  # noqa: E402

RID = re.compile(r"(?P<method>\w+)-(?P<variant>nonoise|noisy_tv|action_noise)-s(?P<seed>\d+)")


def _final(df, col, frac=0.1):
    k = max(1, int(len(df) * frac))
    return df[col].iloc[-k:].mean()


def run(results_dir, positions_dir, figures_dir, n_windows=5):
    os.makedirs(figures_dir, exist_ok=True)
    rows = []
    for path in sorted(glob.glob(os.path.join(results_dir, "*.csv"))):
        m = RID.search(os.path.basename(path))
        if not m:
            continue
        df = pd.read_csv(path)
        if df.empty:
            continue
        rows.append({
            "method": m["method"], "variant": m["variant"], "seed": int(m["seed"]),
            "coverage_frac": _final(df, "coverage_frac"),
            "beyond_wall_frac": _final(df, "beyond_wall_frac"),
            "time_at_wall_frac": _final(df, "time_at_wall_frac"),
            "_df": df,
        })
    if not rows:
        raise SystemExit("no runs found")
    data = pd.DataFrame([{k: v for k, v in r.items() if k != "_df"} for r in rows])

    # --- table ---
    metric_cols = ["coverage_frac", "beyond_wall_frac", "time_at_wall_frac"]
    tbl = data.groupby(["method", "variant"])[metric_cols].agg(["mean", "std"]).round(4)
    tbl.to_csv(os.path.join(figures_dir, "table_coverage.csv"))

    variants = sorted(data["variant"].unique())
    methods = sorted(data["method"].unique())

    # --- coverage curves (one subplot per variant) ---
    fig, axes = plt.subplots(1, len(variants), figsize=(5 * len(variants), 4), squeeze=False)
    for j, v in enumerate(variants):
        ax = axes[0][j]
        for meth in methods:
            curves = [r["_df"] for r in rows if r["method"] == meth and r["variant"] == v]
            if not curves:
                continue
            grid = np.linspace(0, max(c["frames"].max() for c in curves), 100)
            ys = [np.interp(grid, c["frames"], c["coverage_frac"]) for c in curves]
            mean = np.mean(ys, axis=0); std = np.std(ys, axis=0)
            ax.plot(grid, mean, label=meth)
            ax.fill_between(grid, mean - std, mean + std, alpha=0.15)
        ax.set_title(v); ax.set_xlabel("frames"); ax.set_ylabel("coverage_frac")
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(figures_dir, "fig_coverage_curves.png"), dpi=130); plt.close(fig)

    # --- beyond-wall bar (headline robustness) ---
    piv = data.groupby(["method", "variant"])["beyond_wall_frac"].mean().unstack("variant")
    fig, ax = plt.subplots(figsize=(7, 4))
    piv.plot(kind="bar", ax=ax); ax.set_ylabel("beyond_wall_frac (final)")
    ax.set_title("Coverage past the noise wall (room4)")
    fig.tight_layout()
    fig.savefig(os.path.join(figures_dir, "fig_beyond_wall.png"), dpi=130); plt.close(fig)

    # --- time-at-wall bar ---
    piv2 = data.groupby(["method", "variant"])["time_at_wall_frac"].mean().unstack("variant")
    fig, ax = plt.subplots(figsize=(7, 4))
    piv2.plot(kind="bar", ax=ax); ax.set_ylabel("time_at_wall_frac")
    ax.set_title("Fraction of steps lingering at the noise wall")
    fig.tight_layout()
    fig.savefig(os.path.join(figures_dir, "fig_time_at_wall.png"), dpi=130); plt.close(fig)

    # --- noisy-TV fixation (action_noise only): share of steps spent on the
    #     "stare at the TV" action 4. The pure pixel-error method should fixate. ---
    tv_rows = []
    for r in rows:
        if r["variant"] != "action_noise":
            continue
        npz = os.path.join(positions_dir, f"{r['method']}-{r['variant']}-s{r['seed']}.npz")
        if not os.path.exists(npz):
            continue
        a = np.load(npz)["action"][1:]  # drop the initial -1 sentinel
        tv_rows.append({"method": r["method"],
                        "tv_share": float((a == 4).mean()) if len(a) else 0.0})
    if tv_rows:
        tdf = pd.DataFrame(tv_rows).groupby("method")["tv_share"].agg(["mean", "std"])
        tdf.to_csv(os.path.join(figures_dir, "table_tv_fixation.csv"))
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(tdf.index, tdf["mean"], yerr=tdf["std"].fillna(0), capsize=4)
        ax.axhline(0.2, color="gray", ls=":", label="uniform (1/5)")
        ax.set_ylabel("share of steps = action 4 (noisy TV)")
        ax.set_title("Noisy-TV fixation under action_noise")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(figures_dir, "fig_tv_fixation.png"), dpi=130); plt.close(fig)

    # --- heatmap evolution (one fig per variant per mode), averaged over all seeds ---
    for v in variants:
        per_method = {}
        for meth in methods:
            seeds = sorted(c["seed"] for c in rows
                           if c["method"] == meth and c["variant"] == v)
            runs = []
            for seed in seeds:
                npz = os.path.join(positions_dir, f"{meth}-{v}-s{seed}.npz")
                if not os.path.exists(npz):
                    continue
                d = np.load(npz)
                runs.append((d["step"], d["x"], d["z"]))
            if runs:
                per_method[meth] = runs
        if not per_method:
            continue
        for mode in ("density", "frontier"):
            out = os.path.join(figures_dir, f"fig_heatmap_evolution_{v}_{mode}.png")
            heatmaps.plot_evolution(per_method, v, out, mode=mode, n_windows=n_windows)

    print(f"[analyze] wrote table + figures to {figures_dir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--positions", default="positions")
    ap.add_argument("--figures", default="figures")
    ap.add_argument("--n-windows", type=int, default=5)
    a = ap.parse_args()
    run(a.results, a.positions, a.figures, a.n_windows)


if __name__ == "__main__":
    main()
