"""Plot helpers for first experiment metrics."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from evogrid.evaluation.metrics_schema import METRIC_COLUMNS


def require_matplotlib():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError("matplotlib is required for plotting.") from exc
    return plt


def load_rows(csv_path: str | Path) -> list[dict]:
    with Path(csv_path).open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def resolve_metrics_csv(summary_path: str | Path, metrics_csv: str | Path | None = None) -> Path:
    if metrics_csv is not None:
        return Path(metrics_csv)
    summary_path = Path(summary_path)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    metrics_dir = summary.get("outputs", {}).get("metrics_dir")
    if metrics_dir:
        candidate = Path(metrics_dir) / "all_eval.csv"
        if candidate.exists():
            return candidate
    candidate = summary_path.parent / "metrics" / "all_eval.csv"
    if candidate.exists():
        return candidate
    raise FileNotFoundError(f"Could not locate all_eval.csv for {summary_path}")


def summarize_rows(rows: list[dict], metrics: list[str] | None = None) -> dict[str, dict[str, dict[str, float]]]:
    metrics = metrics or METRIC_COLUMNS
    groups = _ordered_groups(rows)
    summary: dict[str, dict[str, dict[str, float]]] = {}
    for group in groups:
        group_rows = [row for row in rows if row.get("group") == group]
        summary[group] = {}
        for metric in metrics:
            values = [_as_float(row.get(metric, 0.0)) for row in group_rows]
            if values:
                mean = sum(values) / len(values)
                variance = sum((value - mean) ** 2 for value in values) / len(values)
                std = variance**0.5
            else:
                mean = 0.0
                std = 0.0
            summary[group][metric] = {"mean": mean, "std": std}
    return summary


def plot_metric_bar(
    summary: dict[str, dict[str, dict[str, float]]],
    metric: str,
    output_path: str | Path,
    title: str,
    ylabel: str,
) -> Path:
    plt = require_matplotlib()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    groups = list(summary.keys())
    means = [summary[group][metric]["mean"] for group in groups]
    stds = [summary[group][metric]["std"] for group in groups]

    fig, ax = plt.subplots(figsize=(8, 4.8))
    colors = ["#3B82F6", "#EF4444", "#94A3B8", "#10B981", "#F59E0B", "#8B5CF6"]
    ax.bar(groups, means, yerr=stds, capsize=4, color=colors[: len(groups)])
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.25)
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def plot_shaping_actions(
    summary: dict[str, dict[str, dict[str, float]]],
    output_path: str | Path,
) -> Path:
    plt = require_matplotlib()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    groups = list(summary.keys())
    dig_means = [summary[group]["num_dig"]["mean"] for group in groups]
    road_means = [summary[group]["num_build_road"]["mean"] for group in groups]
    x_positions = list(range(len(groups)))
    width = 0.36

    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.bar([x - width / 2 for x in x_positions], road_means, width, label="BUILD_ROAD", color="#3B82F6")
    ax.bar([x + width / 2 for x in x_positions], dig_means, width, label="DIG", color="#F59E0B")
    ax.set_title("Environment Shaping Actions")
    ax.set_ylabel("Mean actions per episode")
    ax.set_xticks(x_positions)
    ax.set_xticklabels(groups, rotation=20)
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def plot_first_experiment(summary_path: str | Path, metrics_csv: str | Path | None = None, output_dir: str | Path | None = None) -> list[Path]:
    summary_path = Path(summary_path)
    csv_path = resolve_metrics_csv(summary_path, metrics_csv)
    rows = load_rows(csv_path)
    summary = summarize_rows(
        rows,
        metrics=[
            "episode_reward",
            "ore_delivered",
            "road_usage_rate",
            "num_build_road",
            "num_dig",
        ],
    )
    if output_dir is None:
        output_dir = summary_path.parent / "figures"
    output_dir = Path(output_dir)
    outputs = [
        plot_metric_bar(summary, "episode_reward", output_dir / "reward_comparison.png", "Episode Reward", "Mean reward"),
        plot_metric_bar(summary, "ore_delivered", output_dir / "ore_delivered_comparison.png", "Ore Delivered", "Mean delivered ore"),
        plot_metric_bar(summary, "road_usage_rate", output_dir / "road_usage_rate_comparison.png", "Road Usage Rate", "Mean road usage rate"),
        plot_shaping_actions(summary, output_dir / "shaping_actions_comparison.png"),
    ]
    return outputs


def _ordered_groups(rows: list[dict]) -> list[str]:
    preferred = ["full_shaping", "no_shaping", "random", "greedy", "deepseek_planner", "hybrid_deepseek_greedy"]
    present = {row.get("group", "") for row in rows}
    ordered = [group for group in preferred if group in present]
    ordered.extend(sorted(group for group in present if group and group not in ordered))
    return ordered


def _as_float(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0
