"""Build publication-quality figures for the completed MiniGrid and MiniWorld runs.

The script reads raw experiment artifacts from ``expr_data`` and writes PNG and
PDF versions to each domain's ``figures/publication`` directory.

Run from the repository root:

    conda run -n syssec_exercise python reports/make_publication_figures.py
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, PercentFormatter
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
MINIGRID_ROOT = REPO_ROOT / "expr_data" / "minigrid"
MINIWORLD_ROOT = REPO_ROOT / "expr_data" / "miniworld"
MINIGRID_OUT = MINIGRID_ROOT / "figures" / "publication"
MINIWORLD_OUT = MINIWORLD_ROOT / "figures" / "publication"

METHODS = ["none", "entropy", "rnd", "icm", "lpm"]
METHOD_LABELS = {
    "none": "PPO",
    "entropy": "PPO + entropy",
    "rnd": "RND",
    "icm": "ICM",
    "lpm": "LPM",
}
METHOD_COLORS = {
    "none": "#4D4D4D",
    "entropy": "#0072B2",
    "rnd": "#E69F00",
    "icm": "#009E73",
    "lpm": "#D55E00",
}
METHOD_MARKERS = {
    "none": "o",
    "entropy": "s",
    "rnd": "^",
    "icm": "D",
    "lpm": "P",
}
METHOD_LINESTYLES = {
    "none": "-",
    "entropy": "--",
    "rnd": "-",
    "icm": "-.",
    "lpm": "-",
}

ENVIRONMENTS = [
    "MiniGrid-DoorKey-5x5-v0",
    "MiniGrid-FourRooms-v0",
    "MiniGrid-MultiRoom-N6-v0",
]
ENV_LABELS = {
    "MiniGrid-DoorKey-5x5-v0": "DoorKey-5x5",
    "MiniGrid-FourRooms-v0": "FourRooms",
    "MiniGrid-MultiRoom-N6-v0": "MultiRoom-N6",
}
THEORETICAL_MAX = {
    "MiniGrid-DoorKey-5x5-v0": 0.965,
    "MiniGrid-FourRooms-v0": 0.856,
    "MiniGrid-MultiRoom-N6-v0": 0.652,
}

LAMBDA_COLORS = {
    0.1: "#0072B2",
    0.3: "#009E73",
    1.0: "#D55E00",
    3.0: "#CC79A7",
}
LAMBDA_MARKERS = {0.1: "o", 0.3: "s", 1.0: "^", 3.0: "D"}


def configure_style() -> None:
    """Apply a restrained, colorblind-friendly paper style."""
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.5,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "axes.titleweight": "semibold",
            "axes.labelcolor": "#262626",
            "axes.edgecolor": "#4D4D4D",
            "axes.linewidth": 0.8,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "xtick.color": "#333333",
            "ytick.color": "#333333",
            "legend.fontsize": 8.5,
            "legend.frameon": False,
            "figure.titlesize": 12.5,
            "figure.titleweight": "semibold",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.bbox": "tight",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def polish_axis(ax: plt.Axes, grid_axis: str = "y") -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis=grid_axis, color="#D9D9D9", linewidth=0.65, alpha=0.7)
    ax.set_axisbelow(True)
    ax.tick_params(length=3, width=0.7)


def save_figure(fig: plt.Figure, out_dir: Path, stem: str) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    outputs = []
    for suffix in ("png", "pdf"):
        path = out_dir / f"{stem}.{suffix}"
        kwargs = {"dpi": 300} if suffix == "png" else {}
        fig.savefig(path, **kwargs)
        outputs.append(path)
    plt.close(fig)
    return outputs


def millions(value: float, _position: int) -> str:
    if abs(value) < 1e-12:
        return "0"
    if abs(value) >= 1_000_000:
        return f"{value / 1_000_000:g}M"
    if abs(value) >= 1_000:
        return f"{value / 1_000:g}k"
    return f"{value:g}"


def mean_sem(values: np.ndarray) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return np.nan, np.nan
    if len(values) == 1:
        return float(values[0]), 0.0
    return float(values.mean()), float(values.std(ddof=1) / np.sqrt(len(values)))


def aggregate_curves(
    frame: pd.DataFrame,
    *,
    x_col: str,
    y_col: str,
    seed_col: str = "seed",
    smooth: int = 1,
    points: int = 240,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Interpolate seed traces to a common grid and return mean and SEM."""
    curves: list[tuple[np.ndarray, np.ndarray]] = []
    for _seed, seed_frame in frame.groupby(seed_col):
        curve = (
            seed_frame[[x_col, y_col]]
            .dropna()
            .groupby(x_col, as_index=False)[y_col]
            .mean()
            .sort_values(x_col)
        )
        if len(curve) < 2:
            continue
        x = curve[x_col].to_numpy(dtype=float)
        y = curve[y_col].to_numpy(dtype=float)
        if smooth > 1:
            y = (
                pd.Series(y)
                .rolling(smooth, center=True, min_periods=1)
                .mean()
                .to_numpy()
            )
        curves.append((x, y))

    if not curves:
        return np.array([]), np.array([]), np.array([]), 0

    start = max(x[0] for x, _y in curves)
    stop = min(x[-1] for x, _y in curves)
    grid = np.linspace(start, stop, points)
    values = np.vstack([np.interp(grid, x, y) for x, y in curves])
    mean = values.mean(axis=0)
    if len(curves) > 1:
        sem = values.std(axis=0, ddof=1) / np.sqrt(len(curves))
    else:
        sem = np.zeros_like(mean)
    return grid, mean, sem, len(curves)


def _variant(method: str, condition: str) -> str:
    prefix = "baseline" if method in {"none", "entropy"} else "intrinsic"
    return f"{prefix}_{condition}"


def load_minigrid() -> tuple[pd.DataFrame, pd.DataFrame]:
    sys.path.insert(0, str(REPO_ROOT / "minigrid_exp"))
    from analyze import aggregate_eval_curves, per_seed_final_returns  # noqa: PLC0415
    import config as minigrid_config  # noqa: PLC0415

    frame = aggregate_eval_curves(str(minigrid_config.LOGS_DIR))
    if frame.empty:
        raise RuntimeError(f"No MiniGrid evaluation data found under {minigrid_config.LOGS_DIR}")
    frame["beta_value"] = pd.to_numeric(frame["beta"], errors="coerce")
    frame["noise_probability"] = pd.to_numeric(frame["np"], errors="coerce")

    finals = per_seed_final_returns(frame)
    finals["beta_value"] = pd.to_numeric(finals["beta"], errors="coerce")
    finals["noise_probability"] = pd.to_numeric(finals["np"], errors="coerce")
    return frame, finals


def default_minigrid_rows(
    frame: pd.DataFrame,
    env: str,
    method: str,
    condition: str,
    noise_probability: float | None = None,
) -> pd.DataFrame:
    rows = frame[
        (frame["env"] == env)
        & (frame["method"] == method)
        & (frame["variant"] == _variant(method, condition))
        & (frame["beta"].isna())
    ]
    if noise_probability is not None:
        rows = rows[np.isclose(rows["noise_probability"], noise_probability)]
    return rows


def method_legend() -> list[Line2D]:
    return [
        Line2D(
            [0],
            [0],
            color=METHOD_COLORS[method],
            marker=METHOD_MARKERS[method],
            linestyle=METHOD_LINESTYLES[method],
            linewidth=2.0,
            markersize=5,
            label=METHOD_LABELS[method],
        )
        for method in METHODS
    ]


def plot_minigrid_clean_curves(frame: pd.DataFrame) -> list[Path]:
    fig, axes = plt.subplots(1, 3, figsize=(12.4, 3.7), sharey=True)
    for ax, env in zip(axes, ENVIRONMENTS):
        for method in METHODS:
            rows = default_minigrid_rows(frame, env, method, "no_noise")
            x, mean, sem, _n = aggregate_curves(
                rows, x_col="timestep", y_col="mean_return", smooth=7
            )
            if len(x) == 0:
                continue
            color = METHOD_COLORS[method]
            ax.plot(
                x,
                mean,
                color=color,
                linestyle=METHOD_LINESTYLES[method],
                linewidth=2.0,
                marker=METHOD_MARKERS[method],
                markevery=32,
                markersize=3.8,
                markeredgecolor="white",
                markeredgewidth=0.45,
            )
            ax.fill_between(x, mean - sem, mean + sem, color=color, alpha=0.14, linewidth=0)
        ax.set_title(ENV_LABELS[env])
        ax.set_xlabel("Environment steps")
        ax.xaxis.set_major_formatter(FuncFormatter(millions))
        ax.set_ylim(-0.02, 1.02)
        polish_axis(ax)
    axes[0].set_ylabel("Evaluation return")
    fig.suptitle("MiniGrid clean learning curves", y=0.99)
    fig.legend(
        handles=method_legend(),
        loc="lower center",
        bbox_to_anchor=(0.5, -0.015),
        ncol=5,
        columnspacing=1.8,
        handlelength=2.5,
    )
    fig.text(0.995, 0.015, "Mean ± s.e.m. across 3 seeds", ha="right", va="bottom", color="#595959", fontsize=8)
    fig.tight_layout(rect=(0, 0.12, 1, 0.94))
    return save_figure(fig, MINIGRID_OUT, "minigrid_clean_learning_curves")


def plot_minigrid_final_performance(finals: pd.DataFrame) -> list[Path]:
    fig, axes = plt.subplots(1, 3, figsize=(12.4, 3.65), sharex=True, sharey=True)
    y = np.arange(len(METHODS))
    offsets = np.linspace(-0.10, 0.10, 3)

    for ax, env in zip(axes, ENVIRONMENTS):
        for index, method in enumerate(METHODS):
            rows = default_minigrid_rows(finals, env, method, "no_noise")
            values = rows["final_return"].to_numpy(dtype=float)
            mean, sem = mean_sem(values)
            if not np.isfinite(mean):
                continue
            dot_offsets = offsets if len(values) == 3 else np.linspace(-0.10, 0.10, len(values))
            ax.scatter(
                values,
                np.full(len(values), y[index]) + dot_offsets,
                s=20,
                color=METHOD_COLORS[method],
                alpha=0.35,
                edgecolors="none",
                zorder=2,
            )
            ax.errorbar(
                mean,
                y[index],
                xerr=sem,
                fmt=METHOD_MARKERS[method],
                color=METHOD_COLORS[method],
                markeredgecolor="white",
                markeredgewidth=0.7,
                markersize=7,
                elinewidth=1.8,
                capsize=3,
                zorder=3,
            )
        ax.axvline(THEORETICAL_MAX[env], color="#8C8C8C", linestyle=":", linewidth=1.2)
        ax.set_title(ENV_LABELS[env])
        ax.set_xlabel("Final evaluation return")
        ax.set_xlim(-0.03, 1.03)
        polish_axis(ax, grid_axis="x")

    axes[0].set_yticks(y)
    axes[0].set_yticklabels([METHOD_LABELS[m] for m in METHODS])
    axes[0].invert_yaxis()
    fig.suptitle("MiniGrid final performance after training", y=0.99)
    legend = [
        Line2D([0], [0], color="#8C8C8C", linestyle=":", label="Optimal-policy ceiling"),
        Line2D([0], [0], marker="o", linestyle="none", color="#777777", alpha=0.4, label="Individual seed"),
        Line2D([0], [0], marker="o", linestyle="-", color="#333333", label="Mean ± s.e.m."),
    ]
    fig.legend(handles=legend, loc="lower center", bbox_to_anchor=(0.5, -0.015), ncol=3)
    fig.tight_layout(rect=(0, 0.12, 1, 0.94))
    return save_figure(fig, MINIGRID_OUT, "minigrid_final_performance")


def plot_doorkey_noise_curves(frame: pd.DataFrame) -> list[Path]:
    env = "MiniGrid-DoorKey-5x5-v0"
    conditions = [("Clean", "no_noise", None), ("10% observation noise", "noise", 0.1)]
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.8), sharex=True, sharey=True)
    for ax, (title, condition, probability) in zip(axes, conditions):
        for method in METHODS:
            rows = default_minigrid_rows(frame, env, method, condition, probability)
            x, mean, sem, _n = aggregate_curves(
                rows, x_col="timestep", y_col="mean_return", smooth=7
            )
            if len(x) == 0:
                continue
            color = METHOD_COLORS[method]
            ax.plot(
                x,
                mean,
                color=color,
                linestyle=METHOD_LINESTYLES[method],
                linewidth=2.0,
                marker=METHOD_MARKERS[method],
                markevery=32,
                markersize=3.8,
                markeredgecolor="white",
                markeredgewidth=0.45,
            )
            ax.fill_between(x, mean - sem, mean + sem, color=color, alpha=0.14, linewidth=0)
        ax.set_title(title)
        ax.set_xlabel("Environment steps")
        ax.xaxis.set_major_formatter(FuncFormatter(millions))
        ax.set_ylim(-0.02, 1.02)
        polish_axis(ax)
    axes[0].set_ylabel("Evaluation return")
    fig.suptitle("DoorKey-5x5 learning under observation noise", y=0.99)
    fig.legend(
        handles=method_legend(),
        loc="lower center",
        bbox_to_anchor=(0.5, -0.015),
        ncol=5,
        columnspacing=1.4,
    )
    fig.text(0.995, 0.015, "Mean ± s.e.m. across 3 seeds", ha="right", va="bottom", color="#595959", fontsize=8)
    fig.tight_layout(rect=(0, 0.13, 1, 0.94))
    return save_figure(fig, MINIGRID_OUT, "minigrid_doorkey_noise_learning_curves")


def _noise_final_rows(finals: pd.DataFrame, env: str, method: str) -> pd.DataFrame:
    noisy = default_minigrid_rows(finals, env, method, "noise")
    noisy = noisy[noisy["noise_probability"].notna()].copy()
    probabilities = set(noisy["noise_probability"].round(10))
    if 0.0 not in probabilities:
        clean = default_minigrid_rows(finals, env, method, "no_noise").copy()
        clean["noise_probability"] = 0.0
        noisy = pd.concat([clean, noisy], ignore_index=True)
    return noisy


def plot_minigrid_noise_robustness(finals: pd.DataFrame) -> list[Path]:
    fig, axes = plt.subplots(1, 3, figsize=(12.4, 3.75), sharey=True)
    for ax, env in zip(axes, ENVIRONMENTS):
        for method in METHODS:
            rows = _noise_final_rows(finals, env, method)
            points = []
            for probability, group in rows.groupby("noise_probability"):
                mean, sem = mean_sem(group["final_return"].to_numpy(dtype=float))
                points.append((float(probability), mean, sem))
            if not points:
                continue
            points.sort()
            probability = np.array([p[0] for p in points])
            mean = np.array([p[1] for p in points])
            sem = np.array([p[2] for p in points])
            ax.errorbar(
                probability,
                mean,
                yerr=sem,
                color=METHOD_COLORS[method],
                marker=METHOD_MARKERS[method],
                linestyle=METHOD_LINESTYLES[method],
                linewidth=1.8,
                markersize=5,
                markeredgecolor="white",
                markeredgewidth=0.5,
                capsize=2.5,
            )
        ax.set_title(ENV_LABELS[env])
        ax.set_xlabel("Corrupted observations")
        ax.xaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
        ax.set_xlim(-0.005, 0.105)
        ax.set_ylim(-0.02, 1.02)
        polish_axis(ax)
    axes[0].set_ylabel("Final evaluation return")
    fig.suptitle("MiniGrid observation-noise robustness", y=0.99)
    fig.legend(
        handles=method_legend(),
        loc="lower center",
        bbox_to_anchor=(0.5, -0.015),
        ncol=5,
        columnspacing=1.8,
    )
    fig.text(0.995, 0.015, "Mean ± s.e.m. across 3 seeds", ha="right", va="bottom", color="#595959", fontsize=8)
    fig.tight_layout(rect=(0, 0.13, 1, 0.94))
    return save_figure(fig, MINIGRID_OUT, "minigrid_noise_robustness")


def plot_minigrid_beta_sweep(finals: pd.DataFrame) -> list[Path]:
    env = "MiniGrid-FourRooms-v0"
    methods = ["rnd", "icm", "lpm"]
    fig, ax = plt.subplots(figsize=(7.2, 4.15))

    for method in methods:
        rows = finals[
            (finals["env"] == env)
            & (finals["variant"] == "intrinsic_no_noise")
            & (finals["method"] == method)
            & (finals["beta"].notna())
        ]
        points = []
        for beta, group in rows.groupby("beta_value"):
            mean, sem = mean_sem(group["final_return"].to_numpy(dtype=float))
            points.append((float(beta), mean, sem))
        points.sort()
        beta = np.array([p[0] for p in points])
        mean = np.array([p[1] for p in points])
        sem = np.array([p[2] for p in points])
        ax.errorbar(
            beta,
            mean,
            yerr=sem,
            label=METHOD_LABELS[method],
            color=METHOD_COLORS[method],
            marker=METHOD_MARKERS[method],
            linewidth=2.0,
            markersize=6,
            markeredgecolor="white",
            markeredgewidth=0.6,
            capsize=3,
        )

    baseline = default_minigrid_rows(finals, env, "none", "no_noise")
    baseline_mean, baseline_sem = mean_sem(baseline["final_return"].to_numpy(dtype=float))
    ax.axhline(baseline_mean, color=METHOD_COLORS["none"], linestyle="--", linewidth=1.5, label="PPO baseline")
    ax.axhspan(
        baseline_mean - baseline_sem,
        baseline_mean + baseline_sem,
        color=METHOD_COLORS["none"],
        alpha=0.09,
        linewidth=0,
    )
    beta_ticks = [0.0, 0.0005, 0.001, 0.005, 0.01, 0.05]
    ax.set_xscale("symlog", linthresh=0.0005, linscale=0.7)
    ax.set_xticks(beta_ticks)
    ax.set_xticklabels(["0", "5e-4", "1e-3", "5e-3", "1e-2", "5e-2"])
    ax.set_xlabel("Intrinsic-reward coefficient β")
    ax.set_ylabel("Final evaluation return")
    ax.set_title("FourRooms intrinsic-reward sensitivity")
    ax.set_xlim(-0.00008, 0.065)
    ax.set_ylim(bottom=-0.01)
    polish_axis(ax)
    ax.legend(ncol=2, loc="upper right")
    fig.text(0.99, 0.012, "Mean ± s.e.m. across 3 seeds", ha="right", va="bottom", color="#595959", fontsize=8)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    return save_figure(fig, MINIGRID_OUT, "minigrid_beta_sensitivity")


def miniworld_paths() -> tuple[Path, Path]:
    sweep_root = MINIWORLD_ROOT / "sweeps"
    screening = sweep_root / "expr1_lpm_core_action_noise"
    validation = sweep_root / "expr2_lpm_top3_action_noise_64seed"
    for path in (screening, validation):
        if not path.exists():
            raise RuntimeError(f"Missing MiniWorld sweep directory: {path}")
    return screening, validation


def read_config_lambda(sweep_root: Path, slug: str) -> float:
    with (sweep_root / "configs" / f"{slug}.json").open(encoding="utf-8") as handle:
        config = json.load(handle)
    return float(config["lambda_intrinsic"])


def load_miniworld_traces(sweep_root: Path) -> pd.DataFrame:
    frames = []
    for config_dir in sorted((sweep_root / "results").iterdir()):
        if not config_dir.is_dir():
            continue
        lambda_value = read_config_lambda(sweep_root, config_dir.name)
        for csv_path in sorted(config_dir.glob("*.csv")):
            match = re.search(r"-s(?P<seed>\d+)$", csv_path.stem)
            if not match:
                continue
            trace = pd.read_csv(csv_path)
            trace["seed"] = int(match.group("seed"))
            trace["lambda_intrinsic"] = lambda_value
            trace["config_slug"] = config_dir.name
            frames.append(trace)
    if not frames:
        raise RuntimeError(f"No MiniWorld traces found under {sweep_root / 'results'}")
    return pd.concat(frames, ignore_index=True)


def plot_miniworld_sensitivity(screening: Path) -> list[Path]:
    summary = pd.read_csv(screening / "summary" / "summary_by_config.csv")
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.85))
    metrics = [
        ("coverage_frac", "Final maze coverage (%)"),
        ("tv_share", "Noise-action share (%)"),
    ]
    entropy_values = sorted(summary["entropy_coef"].unique())
    entropy_colors = ["#7A7A7A", "#0072B2", "#D55E00"]
    entropy_markers = ["o", "s", "^"]

    for ax, (metric, label) in zip(axes, metrics):
        for entropy, color, marker in zip(entropy_values, entropy_colors, entropy_markers):
            rows = summary[np.isclose(summary["entropy_coef"], entropy)].sort_values("lambda_intrinsic")
            x = rows["lambda_intrinsic"].to_numpy(dtype=float)
            mean = 100.0 * rows[f"{metric}_mean"].to_numpy(dtype=float)
            count = rows[f"{metric}_count"].to_numpy(dtype=float)
            sem = 100.0 * rows[f"{metric}_std"].to_numpy(dtype=float) / np.sqrt(count)
            ax.errorbar(
                x,
                mean,
                yerr=sem,
                color=color,
                marker=marker,
                linewidth=1.9,
                markersize=5.5,
                markeredgecolor="white",
                markeredgewidth=0.6,
                capsize=3,
                label=f"Entropy = {entropy:g}",
            )
        ax.set_xscale("log")
        ax.set_xticks([0.1, 0.3, 1.0, 3.0])
        ax.set_xticklabels(["0.1", "0.3", "1", "3"])
        ax.set_xlabel("LPM intrinsic coefficient λ")
        ax.set_ylabel(label)
        ax.set_ylim(bottom=0)
        polish_axis(ax)

    axes[0].set_title("Exploration")
    axes[1].set_title("Noise fixation")
    fig.suptitle("MiniWorld LPM hyperparameter screening", y=0.99)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, -0.015), ncol=3)
    fig.text(0.995, 0.015, "Mean ± s.e.m. across 8 seeds", ha="right", va="bottom", color="#595959", fontsize=8)
    fig.tight_layout(rect=(0, 0.14, 1, 0.94))
    return save_figure(fig, MINIWORLD_OUT, "miniworld_lpm_hyperparameter_sensitivity")


def plot_miniworld_validation_curves(validation: Path) -> list[Path]:
    traces = load_miniworld_traces(validation)
    lambdas = sorted(traces["lambda_intrinsic"].unique())
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.85))
    metrics = [
        ("coverage_frac", "Maze coverage (%)"),
        ("time_at_wall_frac", "Time at noise wall (%)"),
    ]

    for ax, (metric, label) in zip(axes, metrics):
        for lambda_value in lambdas:
            rows = traces[np.isclose(traces["lambda_intrinsic"], lambda_value)]
            x, mean, sem, _n = aggregate_curves(
                rows, x_col="frames", y_col=metric, smooth=1, points=200
            )
            mean = 100.0 * mean
            sem = 100.0 * sem
            color = LAMBDA_COLORS[float(lambda_value)]
            ax.plot(
                x,
                mean,
                color=color,
                linewidth=2.1,
                marker=LAMBDA_MARKERS[float(lambda_value)],
                markevery=32,
                markersize=3.8,
                markeredgecolor="white",
                markeredgewidth=0.45,
                label=f"λ = {lambda_value:g}",
            )
            ax.fill_between(x, mean - sem, mean + sem, color=color, alpha=0.15, linewidth=0)
        ax.set_xlabel("Environment steps")
        ax.xaxis.set_major_formatter(FuncFormatter(millions))
        ax.set_ylabel(label)
        ax.set_ylim(bottom=0)
        polish_axis(ax)

    axes[0].set_title("Exploration")
    axes[1].set_title("Noise exposure")
    fig.suptitle("MiniWorld 64-seed validation learning curves", y=0.99)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, -0.015), ncol=3)
    fig.text(0.995, 0.015, "Mean ± s.e.m. across 64 seeds", ha="right", va="bottom", color="#595959", fontsize=8)
    fig.tight_layout(rect=(0, 0.14, 1, 0.94))
    return save_figure(fig, MINIWORLD_OUT, "miniworld_validation_learning_curves")


def _draw_violin_distribution(
    ax: plt.Axes,
    values: np.ndarray,
    position: float,
    color: str,
    rng: np.random.Generator,
) -> None:
    parts = ax.violinplot(
        values,
        positions=[position],
        widths=0.72,
        showmeans=False,
        showmedians=False,
        showextrema=False,
    )
    for body in parts["bodies"]:
        body.set_facecolor(color)
        body.set_edgecolor(color)
        body.set_alpha(0.16)
        body.set_linewidth(0.8)

    jitter = np.clip(rng.normal(0.0, 0.075, len(values)), -0.19, 0.19)
    ax.scatter(
        np.full(len(values), position) + jitter,
        values,
        s=14,
        color=color,
        alpha=0.32,
        edgecolors="none",
        zorder=2,
    )
    q1, median, q3 = np.percentile(values, [25, 50, 75])
    ax.vlines(position, q1, q3, color=color, linewidth=4.5, zorder=3)
    ax.scatter(position, median, marker="_", s=100, color="white", linewidth=1.4, zorder=4)
    mean, sem = mean_sem(values)
    ax.errorbar(
        position,
        mean,
        yerr=1.96 * sem,
        fmt="D",
        color=color,
        markeredgecolor="white",
        markeredgewidth=0.7,
        markersize=5.5,
        elinewidth=1.6,
        capsize=3,
        zorder=5,
    )


def plot_miniworld_validation_distributions(validation: Path) -> list[Path]:
    by_run = pd.read_csv(validation / "summary" / "summary_by_run.csv")
    lambdas = sorted(by_run["lambda_intrinsic"].unique())
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.85))
    metrics = [
        ("coverage_frac", "Final maze coverage (%)"),
        ("tv_share", "Noise-action share (%)"),
    ]
    rng = np.random.default_rng(20260711)

    for ax, (metric, label) in zip(axes, metrics):
        for position, lambda_value in enumerate(lambdas, start=1):
            values = 100.0 * by_run.loc[
                np.isclose(by_run["lambda_intrinsic"], lambda_value), metric
            ].dropna().to_numpy(dtype=float)
            _draw_violin_distribution(
                ax,
                values,
                position,
                LAMBDA_COLORS[float(lambda_value)],
                rng,
            )
        ax.set_xticks(range(1, len(lambdas) + 1))
        ax.set_xticklabels([f"λ = {value:g}" for value in lambdas])
        ax.set_ylabel(label)
        if metric == "tv_share":
            # Preserve the rare 100% fixation failures without flattening the
            # majority of seeds concentrated below 10%.
            ax.set_yscale("symlog", linthresh=1.0, linscale=1.0)
            ax.set_yticks([0, 1, 5, 10, 50, 100])
            ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _pos: f"{value:g}"))
            ax.set_ylabel("Noise-action share (%, symlog scale)")
        ax.set_ylim(bottom=0)
        polish_axis(ax)

    axes[0].set_title("Exploration")
    axes[1].set_title("Noise fixation")
    fig.suptitle("MiniWorld validation across random seeds", y=0.99)
    legend = [
        Line2D([0], [0], marker="o", linestyle="none", color="#666666", alpha=0.4, label="Seed"),
        Line2D([0], [0], linewidth=5, color="#666666", label="Interquartile range"),
        Line2D([0], [0], marker="D", linestyle="-", color="#666666", label="Mean ± 95% CI"),
    ]
    fig.legend(handles=legend, loc="lower center", bbox_to_anchor=(0.5, -0.015), ncol=3)
    fig.tight_layout(rect=(0, 0.14, 1, 0.94))
    return save_figure(fig, MINIWORLD_OUT, "miniworld_validation_seed_distributions")


def plot_miniworld_screen_vs_validation(screening: Path, validation: Path) -> list[Path]:
    screen = pd.read_csv(screening / "summary" / "summary_by_config.csv")
    valid = pd.read_csv(validation / "summary" / "summary_by_config.csv")
    candidates = sorted(valid["lambda_intrinsic"].unique())
    screen = screen[
        screen["lambda_intrinsic"].isin(candidates)
        & np.isclose(screen["entropy_coef"], 0.05)
    ]

    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.85))
    metrics = [
        ("coverage_frac", "Final maze coverage (%)"),
        ("tv_share", "Noise-action share (%)"),
    ]
    stages = [("8-seed screen", screen), ("64-seed validation", valid)]

    for ax, (metric, label) in zip(axes, metrics):
        for lambda_value in candidates:
            means = []
            errors = []
            for _stage_name, data in stages:
                row = data[np.isclose(data["lambda_intrinsic"], lambda_value)].iloc[0]
                means.append(100.0 * float(row[f"{metric}_mean"]))
                errors.append(
                    100.0
                    * float(row[f"{metric}_std"])
                    / np.sqrt(float(row[f"{metric}_count"]))
                )
            color = LAMBDA_COLORS[float(lambda_value)]
            ax.errorbar(
                [0, 1],
                means,
                yerr=errors,
                color=color,
                marker=LAMBDA_MARKERS[float(lambda_value)],
                linewidth=2.0,
                markersize=6,
                markeredgecolor="white",
                markeredgewidth=0.6,
                capsize=3,
                label=f"λ = {lambda_value:g}",
            )
        ax.set_xticks([0, 1])
        ax.set_xticklabels([stage[0] for stage in stages])
        ax.set_xlim(-0.18, 1.18)
        ax.set_ylabel(label)
        ax.set_ylim(bottom=0)
        polish_axis(ax)

    axes[0].set_title("Exploration")
    axes[1].set_title("Noise fixation")
    fig.suptitle("MiniWorld screening result versus held-out validation", y=0.99)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, -0.015), ncol=3)
    fig.text(0.995, 0.015, "Mean ± s.e.m.", ha="right", va="bottom", color="#595959", fontsize=8)
    fig.tight_layout(rect=(0, 0.14, 1, 0.94))
    return save_figure(fig, MINIWORLD_OUT, "miniworld_screen_vs_validation")


def write_manifest(out_dir: Path, title: str, descriptions: list[tuple[str, str]]) -> None:
    lines = [f"# {title}", "", "Every figure is available as a 300-DPI PNG and a vector PDF.", ""]
    for stem, description in descriptions:
        lines.append(f"- `{stem}.png` / `{stem}.pdf` - {description}")
    lines.extend(
        [
            "",
            "Regenerate from the repository root:",
            "",
            "```powershell",
            "conda run -n syssec_exercise python reports/make_publication_figures.py",
            "```",
            "",
        ]
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")


def build_minigrid() -> list[Path]:
    frame, finals = load_minigrid()
    outputs = []
    outputs.extend(plot_minigrid_clean_curves(frame))
    outputs.extend(plot_minigrid_final_performance(finals))
    outputs.extend(plot_doorkey_noise_curves(frame))
    outputs.extend(plot_minigrid_noise_robustness(finals))
    outputs.extend(plot_minigrid_beta_sweep(finals))
    write_manifest(
        MINIGRID_OUT,
        "MiniGrid publication figures",
        [
            ("minigrid_clean_learning_curves", "Sample efficiency for all methods on the difficulty ladder."),
            ("minigrid_final_performance", "Per-seed final returns, means, and uncertainty."),
            ("minigrid_doorkey_noise_learning_curves", "Clean and noisy DoorKey learning dynamics."),
            ("minigrid_noise_robustness", "Final return across observation-noise probabilities."),
            ("minigrid_beta_sensitivity", "FourRooms intrinsic-reward coefficient sweep."),
        ],
    )
    return outputs


def build_miniworld() -> list[Path]:
    screening, validation = miniworld_paths()
    outputs = []
    outputs.extend(plot_miniworld_sensitivity(screening))
    outputs.extend(plot_miniworld_validation_curves(validation))
    outputs.extend(plot_miniworld_validation_distributions(validation))
    outputs.extend(plot_miniworld_screen_vs_validation(screening, validation))
    write_manifest(
        MINIWORLD_OUT,
        "MiniWorld publication figures",
        [
            ("miniworld_lpm_hyperparameter_sensitivity", "Eight-seed λ and entropy screening sweep."),
            ("miniworld_validation_learning_curves", "Coverage and noise exposure over 50,000 steps."),
            ("miniworld_validation_seed_distributions", "Full 64-seed final-metric distributions."),
            ("miniworld_screen_vs_validation", "Screening estimates compared with held-out validation."),
        ],
    )
    return outputs


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--domain",
        choices=("all", "minigrid", "miniworld"),
        default="all",
        help="figure group to regenerate",
    )
    args = parser.parse_args(argv)
    configure_style()

    outputs: list[Path] = []
    if args.domain in {"all", "minigrid"}:
        outputs.extend(build_minigrid())
    if args.domain in {"all", "miniworld"}:
        outputs.extend(build_miniworld())

    print(f"Wrote {len(outputs)} figure files")
    for output in outputs:
        print(output.relative_to(REPO_ROOT))


if __name__ == "__main__":
    main()
