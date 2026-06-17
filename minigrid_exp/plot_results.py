from __future__ import annotations

import csv
from html import escape
from pathlib import Path

from config import ALGORITHM_LABEL, EVAL_RESULTS_PATH, EVAL_SUMMARY_PATH, PLOTS_DIR


SUMMARY_PATH = EVAL_SUMMARY_PATH
RESULTS_PATH = EVAL_RESULTS_PATH

VARIANT_ORDER = [
    "baseline_no_noise",
    "baseline_noise",
    "intrinsic_no_noise",
    "intrinsic_noise",
]

VARIANT_LABELS = {
    "baseline_no_noise": "Baseline",
    "baseline_noise": "Baseline + noise",
    "intrinsic_no_noise": "Intrinsic",
    "intrinsic_noise": "Intrinsic + noise",
}

VARIANT_COLORS = {
    "baseline_no_noise": "#3b6fb6",
    "baseline_noise": "#d97c2b",
    "intrinsic_no_noise": "#4c9a68",
    "intrinsic_noise": "#8d5fbf",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as csv_file:
        return list(csv.DictReader(csv_file))


def as_float(row: dict[str, str], key: str) -> float:
    value = row.get(key, "")
    return float(value) if value != "" else 0.0


def unique_in_order(rows: list[dict[str, str]], key: str) -> list[str]:
    seen = set()
    values = []
    for row in rows:
        value = row[key]
        if value not in seen:
            seen.add(value)
            values.append(value)
    return values


def ordered_variants(rows: list[dict[str, str]]) -> list[str]:
    present = unique_in_order(rows, "variant")
    configured = [variant for variant in VARIANT_ORDER if variant in present]
    unexpected = sorted(variant for variant in present if variant not in configured)
    return configured + unexpected


def format_env_name(env_id: str) -> str:
    return env_id.replace("MiniGrid-", "").replace("-v0", "")


def svg_text(
    x: float,
    y: float,
    text: str,
    size: int = 13,
    anchor: str = "middle",
    weight: str = "400",
    rotate: float | None = None,
) -> str:
    transform = f' transform="rotate({rotate} {x:.1f} {y:.1f})"' if rotate else ""
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" '
        f'font-family="Arial, sans-serif" font-weight="{weight}" '
        f'text-anchor="{anchor}" fill="#222222"{transform}>{escape(text)}</text>'
    )


def grouped_bar_plot(
    summary: list[dict[str, str]],
    metric: str,
    error_metric: str,
    ylabel: str,
    title: str,
    output_path: Path,
    y_max: float | None = None,
):
    envs = unique_in_order(summary, "env_id")
    variants = ordered_variants(summary)
    by_key = {(row["env_id"], row["variant"]): row for row in summary}

    values = [as_float(row, metric) + as_float(row, error_metric) for row in summary]
    upper = y_max if y_max is not None else max(values + [1.0]) * 1.15
    upper = max(upper, 1.0)

    width = max(980, 160 * len(envs))
    height = 620
    left = 90
    right = 40
    top = 70
    bottom = 135
    plot_width = width - left - right
    plot_height = height - top - bottom

    def sx(env_index: int, variant_index: int) -> float:
        group_width = plot_width / len(envs)
        bar_width = min(24, group_width * 0.72 / max(len(variants), 1))
        start = left + env_index * group_width + group_width / 2
        offset = (variant_index - (len(variants) - 1) / 2) * (bar_width + 5)
        return start + offset - bar_width / 2

    def sy(value: float) -> float:
        return top + plot_height - (value / upper) * plot_height

    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        svg_text(width / 2, 35, title, size=22, weight="700"),
        svg_text(28, top + plot_height / 2, ylabel, size=14, rotate=-90),
    ]

    for tick in range(6):
        value = upper * tick / 5
        y = sy(value)
        elements.append(f'<line x1="{left}" y1="{y:.1f}" x2="{width - right}" y2="{y:.1f}" stroke="#d8dde6" stroke-width="1"/>')
        elements.append(svg_text(left - 12, y + 4, f"{value:.2f}", size=12, anchor="end"))

    bar_width = min(24, (plot_width / len(envs)) * 0.72 / max(len(variants), 1))
    for env_index, env in enumerate(envs):
        group_center = left + env_index * (plot_width / len(envs)) + (plot_width / len(envs)) / 2
        elements.append(svg_text(group_center, height - 48, format_env_name(env), size=12, rotate=-25))

        for variant_index, variant in enumerate(variants):
            row = by_key.get((env, variant))
            if row is None:
                continue
            value = as_float(row, metric)
            error = as_float(row, error_metric)
            x = sx(env_index, variant_index)
            y = sy(value)
            bar_height = top + plot_height - y
            color = VARIANT_COLORS.get(variant, "#777777")
            elements.append(
                f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_width:.1f}" height="{bar_height:.1f}" '
                f'fill="{color}" stroke="#222222" stroke-width="0.6">'
                f'<title>{escape(env)} / {escape(VARIANT_LABELS.get(variant, variant))}: {value:.3f}</title></rect>'
            )
            if error > 0:
                y_low = sy(max(value - error, 0))
                y_high = sy(value + error)
                cx = x + bar_width / 2
                elements.append(f'<line x1="{cx:.1f}" y1="{y_high:.1f}" x2="{cx:.1f}" y2="{y_low:.1f}" stroke="#222222" stroke-width="1.2"/>')
                elements.append(f'<line x1="{cx - 5:.1f}" y1="{y_high:.1f}" x2="{cx + 5:.1f}" y2="{y_high:.1f}" stroke="#222222" stroke-width="1.2"/>')
                elements.append(f'<line x1="{cx - 5:.1f}" y1="{y_low:.1f}" x2="{cx + 5:.1f}" y2="{y_low:.1f}" stroke="#222222" stroke-width="1.2"/>')

    legend_x = left
    legend_y = height - 20
    for index, variant in enumerate(variants):
        x = legend_x + index * 180
        elements.append(f'<rect x="{x}" y="{legend_y - 12}" width="14" height="14" fill="{VARIANT_COLORS.get(variant, "#777777")}" stroke="#222222" stroke-width="0.5"/>')
        elements.append(svg_text(x + 20, legend_y, VARIANT_LABELS.get(variant, variant), size=12, anchor="start"))

    elements.append("</svg>")
    output_path.write_text("\n".join(elements), encoding="utf-8")


def seed_scatter_plot(results: list[dict[str, str]], output_path: Path):
    envs = unique_in_order(results, "env_id")
    variants = ordered_variants(results)

    width = max(980, 160 * len(envs))
    height = 600
    left = 90
    right = 40
    top = 70
    bottom = 125
    plot_width = width - left - right
    plot_height = height - top - bottom

    def sy(value: float) -> float:
        return top + plot_height - value * plot_height

    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        svg_text(width / 2, 35, f"{ALGORITHM_LABEL} Evaluation Success Rate by Seed", size=22, weight="700"),
        svg_text(28, top + plot_height / 2, "Success rate", size=14, rotate=-90),
    ]

    for tick in range(6):
        value = tick / 5
        y = sy(value)
        elements.append(f'<line x1="{left}" y1="{y:.1f}" x2="{width - right}" y2="{y:.1f}" stroke="#d8dde6" stroke-width="1"/>')
        elements.append(svg_text(left - 12, y + 4, f"{value:.1f}", size=12, anchor="end"))

    group_width = plot_width / len(envs)
    offsets = [-0.28, -0.09, 0.09, 0.28]
    if len(variants) != 4:
        offsets = [0.0] if len(variants) == 1 else [(-0.3 + 0.6 * i / (len(variants) - 1)) for i in range(len(variants))]

    for env_index, env in enumerate(envs):
        group_center = left + env_index * group_width + group_width / 2
        elements.append(svg_text(group_center, height - 42, format_env_name(env), size=12, rotate=-25))

        for variant_index, variant in enumerate(variants):
            rows = [row for row in results if row["env_id"] == env and row["variant"] == variant]
            x = group_center + offsets[variant_index] * group_width
            for row in rows:
                value = as_float(row, "success_rate")
                seed = row.get("seed", "?")
                y = sy(value)
                color = VARIANT_COLORS.get(variant, "#777777")
                elements.append(
                    f'<circle cx="{x:.1f}" cy="{y:.1f}" r="6" fill="{color}" stroke="#222222" stroke-width="0.7" opacity="0.88">'
                    f'<title>{escape(env)} / {escape(VARIANT_LABELS.get(variant, variant))} / seed {escape(seed)}: {value:.3f}</title></circle>'
                )

    legend_x = left
    legend_y = height - 18
    for index, variant in enumerate(variants):
        x = legend_x + index * 180
        elements.append(f'<circle cx="{x + 7}" cy="{legend_y - 6}" r="6" fill="{VARIANT_COLORS.get(variant, "#777777")}" stroke="#222222" stroke-width="0.5"/>')
        elements.append(svg_text(x + 20, legend_y, VARIANT_LABELS.get(variant, variant), size=12, anchor="start"))

    elements.append("</svg>")
    output_path.write_text("\n".join(elements), encoding="utf-8")


def write_index(plot_paths: list[Path]):
    links = "\n".join(
        f'<li><a href="{escape(path.name)}">{escape(path.stem.replace("_", " ").title())}</a></li>'
        for path in plot_paths
    )
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>MiniGrid {ALGORITHM_LABEL} Evaluation Plots</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 32px; color: #222; }}
    iframe {{ display: block; width: 100%; height: 660px; border: 1px solid #d8dde6; margin: 20px 0 36px; }}
  </style>
</head>
<body>
  <h1>MiniGrid {ALGORITHM_LABEL} Evaluation Plots</h1>
  <ul>{links}</ul>
  {''.join(f'<iframe src="{escape(path.name)}"></iframe>' for path in plot_paths)}
</body>
</html>
"""
    (PLOTS_DIR / "index.html").write_text(html, encoding="utf-8")


def main():
    if not SUMMARY_PATH.exists():
        raise FileNotFoundError(f"Missing summary CSV: {SUMMARY_PATH}")
    if not RESULTS_PATH.exists():
        raise FileNotFoundError(f"Missing per-seed CSV: {RESULTS_PATH}")

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    summary = sorted(read_csv(SUMMARY_PATH), key=lambda row: (row["env_id"], row["variant"]))
    results = sorted(read_csv(RESULTS_PATH), key=lambda row: (row["env_id"], row["variant"], int(row["seed"])))

    reward_values = [as_float(row, "mean_reward") + as_float(row, "std_reward") for row in summary]
    length_values = [as_float(row, "mean_episode_length") + as_float(row, "std_episode_length") for row in summary]

    plot_paths = [
        PLOTS_DIR / "evaluation_success_rate.svg",
        PLOTS_DIR / "evaluation_reward.svg",
        PLOTS_DIR / "evaluation_episode_length.svg",
        PLOTS_DIR / "evaluation_success_by_seed.svg",
    ]

    grouped_bar_plot(
        summary,
        metric="mean_success_rate",
        error_metric="std_success_rate",
        ylabel="Mean success rate",
        title=f"{ALGORITHM_LABEL} Evaluation Success Rate",
        output_path=plot_paths[0],
        y_max=1.05,
    )
    grouped_bar_plot(
        summary,
        metric="mean_reward",
        error_metric="std_reward",
        ylabel="Mean episode reward",
        title=f"{ALGORITHM_LABEL} Evaluation Reward",
        output_path=plot_paths[1],
        y_max=max(reward_values + [1.0]) * 1.1,
    )
    grouped_bar_plot(
        summary,
        metric="mean_episode_length",
        error_metric="std_episode_length",
        ylabel="Mean episode length",
        title=f"{ALGORITHM_LABEL} Evaluation Episode Length",
        output_path=plot_paths[2],
        y_max=max(length_values + [1.0]) * 1.1,
    )
    seed_scatter_plot(results, plot_paths[3])
    write_index(plot_paths)

    print(f"Saved {len(plot_paths)} plots and index.html to: {PLOTS_DIR}")


if __name__ == "__main__":
    main()
