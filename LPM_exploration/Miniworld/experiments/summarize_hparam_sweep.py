"""Summarize a MiniWorld hyperparameter sweep.

Reads the directory layout produced by `run_hparam_sweep.py` and writes:

- `summary_by_run.csv`: final metrics for every completed seed.
- `summary_by_config.csv`: mean/std/count over seeds for each config and variant.

The final metric for a CSV trace is the mean over the last `--final-frac` share
of logged rows, matching the existing maze `analyze.py` convention.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re

import numpy as np
import pandas as pd


RID = re.compile(r"(?P<method>\w+)-(?P<variant>nonoise|noisy_tv|action_noise)-s(?P<seed>\d+)")
METRICS = [
    "coverage_frac",
    "beyond_wall_frac",
    "time_at_wall_frac",
    "int_rew_mean",
    "pred_loss",
    "unc_loss",
    "fwd_loss",
    "rnd_loss",
    "policy_loss",
    "value_loss",
    "entropy",
    "tv_share",
]


def _final(df, col, frac):
    if col not in df or df[col].dropna().empty:
        return np.nan
    series = pd.to_numeric(df[col], errors="coerce").dropna()
    if series.empty:
        return np.nan
    k = max(1, int(len(series) * frac))
    return float(series.iloc[-k:].mean())


def _load_config(sweep_root, config_slug):
    path = os.path.join(sweep_root, "configs", config_slug + ".json")
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _tv_share(sweep_root, config_slug, rid, variant):
    if variant != "action_noise":
        return np.nan
    path = os.path.join(sweep_root, "positions", config_slug, rid + ".npz")
    if not os.path.exists(path):
        return np.nan
    data = np.load(path)
    actions = data["action"][1:]
    if len(actions) == 0:
        return np.nan
    return float((actions == 4).mean())


def summarize(sweep_root, out_dir, final_frac):
    rows = []
    pattern = os.path.join(sweep_root, "results", "*", "*.csv")
    for csv_path in sorted(glob.glob(pattern)):
        config_slug = os.path.basename(os.path.dirname(csv_path))
        rid = os.path.splitext(os.path.basename(csv_path))[0]
        m = RID.fullmatch(rid)
        if not m:
            continue
        df = pd.read_csv(csv_path)
        if df.empty:
            continue
        config = _load_config(sweep_root, config_slug)
        row = {
            "config_slug": config_slug,
            "run_id": rid,
            "method": m["method"],
            "variant": m["variant"],
            "seed": int(m["seed"]),
            **config,
        }
        for metric in METRICS:
            if metric == "tv_share":
                row[metric] = _tv_share(sweep_root, config_slug, rid, m["variant"])
            else:
                row[metric] = _final(df, metric, final_frac)
        rows.append(row)

    if not rows:
        raise SystemExit(f"no completed run CSVs found under {sweep_root}")

    os.makedirs(out_dir, exist_ok=True)
    by_run = pd.DataFrame(rows)
    by_run.to_csv(os.path.join(out_dir, "summary_by_run.csv"), index=False)

    config_cols = [
        c for c in by_run.columns
        if c not in {"run_id", "seed", *METRICS}
    ]
    agg = by_run.groupby(config_cols, dropna=False)[METRICS].agg(["mean", "std", "count"])
    agg.columns = ["_".join(c for c in col if c) for col in agg.columns.to_flat_index()]
    agg = agg.reset_index()
    agg.to_csv(os.path.join(out_dir, "summary_by_config.csv"), index=False)

    ranking_cols = ["config_slug", "method", "variant",
                    "coverage_frac_mean", "tv_share_mean", "time_at_wall_frac_mean"]
    existing = [c for c in ranking_cols if c in agg.columns]
    leaderboard = agg.sort_values(
        by=[c for c in ["coverage_frac_mean", "tv_share_mean"] if c in agg.columns],
        ascending=[False, True][:len([c for c in ["coverage_frac_mean", "tv_share_mean"] if c in agg.columns])],
    )
    leaderboard[existing].to_csv(os.path.join(out_dir, "leaderboard.csv"), index=False)
    print(f"[summarize] wrote {len(by_run)} run rows and {len(agg)} config rows to {out_dir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep-root", required=True)
    ap.add_argument("--out-dir",
                    help="default: <sweep-root>/summary")
    ap.add_argument("--final-frac", type=float, default=0.1,
                    help="share of logged rows used for final metrics")
    args = ap.parse_args()
    if not (0 < args.final_frac <= 1):
        raise SystemExit("--final-frac must be in (0, 1]")
    sweep_root = os.path.abspath(args.sweep_root)
    out_dir = os.path.abspath(args.out_dir or os.path.join(sweep_root, "summary"))
    summarize(sweep_root, out_dir, args.final_frac)


if __name__ == "__main__":
    main()
