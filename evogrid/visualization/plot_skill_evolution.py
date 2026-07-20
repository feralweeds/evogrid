"""Plots for saved Skill-evolution experiment outputs."""

from __future__ import annotations

import csv
from pathlib import Path

from evogrid.visualization.plot_curves import require_matplotlib


def plot_skill_evolution(run_dir: str | Path, output_dir: str | Path | None = None) -> list[Path]:
    run_dir = Path(run_dir)
    output_dir = Path(output_dir or run_dir / "figures")
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoints = _read_csv(run_dir / "capability" / "checkpoints.csv")
    matrix = _read_csv(run_dir / "capability" / "capability_matrix.csv")
    return [
        _plot_verified_skill_count(checkpoints, output_dir / "verified_skill_count.png"),
        _plot_capability_score(checkpoints, output_dir / "capability_score.png"),
        _plot_coverage_heatmap(matrix, output_dir / "skill_coverage_heatmap.png"),
        _plot_false_trigger_retention(checkpoints, output_dir / "false_trigger_retention.png"),
    ]


def _plot_verified_skill_count(rows: list[dict[str, str]], output_path: Path) -> Path:
    plt = require_matplotlib()
    x = list(range(len(rows)))
    counts = [_as_float(row.get("verified_skill_count")) for row in rows]
    labels = [row.get("group", str(index)) for index, row in enumerate(rows)]
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    ax.step(x, counts, where="post", color="#2563EB", linewidth=2.5)
    ax.scatter(x, counts, color="#1D4ED8", s=36)
    _annotate_promotions(ax, x, counts, rows)
    ax.set_title("Verified Skill Count")
    ax.set_ylabel("Count")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=18, ha="right")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def _plot_capability_score(rows: list[dict[str, str]], output_path: Path) -> Path:
    plt = require_matplotlib()
    x = list(range(len(rows)))
    scores = [_as_float(row.get("capability_score")) for row in rows]
    ci_low = [_as_float(row.get("ci_low", score)) for row, score in zip(rows, scores)]
    ci_high = [_as_float(row.get("ci_high", score)) for row, score in zip(rows, scores)]
    labels = [row.get("group", str(index)) for index, row in enumerate(rows)]
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    ax.plot(x, scores, color="#059669", marker="o", linewidth=2.5)
    ax.fill_between(x, ci_low, ci_high, color="#A7F3D0", alpha=0.55)
    _annotate_promotions(ax, x, scores, rows)
    ax.set_title("Capability Score")
    ax.set_ylabel("Score")
    ax.set_ylim(0.0, 1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=18, ha="right")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def _plot_coverage_heatmap(rows: list[dict[str, str]], output_path: Path) -> Path:
    plt = require_matplotlib()
    task_ids = [key for key in rows[0].keys() if key not in {"group", "skill_id"}] if rows else []
    labels = [f"{row.get('group')}:{row.get('skill_id')}" for row in rows]
    values = [[_as_float(row.get(task_id)) for task_id in task_ids] for row in rows]
    fig, ax = plt.subplots(figsize=(8.5, max(3.5, len(labels) * 0.42)))
    image = ax.imshow(values or [[0.0]], aspect="auto", cmap="viridis", vmin=0.0, vmax=1.0)
    ax.set_title("Skill Coverage by Environment")
    ax.set_xticks(list(range(len(task_ids))))
    ax.set_xticklabels(task_ids, rotation=20, ha="right")
    ax.set_yticks(list(range(len(labels))))
    ax.set_yticklabels(labels)
    fig.colorbar(image, ax=ax, fraction=0.04, pad=0.03)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def _plot_false_trigger_retention(rows: list[dict[str, str]], output_path: Path) -> Path:
    plt = require_matplotlib()
    x = list(range(len(rows)))
    false_trigger = [_as_float(row.get("false_trigger_rate")) for row in rows]
    retention = [_as_float(row.get("retention_score")) for row in rows]
    labels = [row.get("group", str(index)) for index, row in enumerate(rows)]
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    ax.plot(x, false_trigger, marker="o", color="#DC2626", linewidth=2.0, label="False trigger")
    ax.plot(x, retention, marker="s", color="#7C3AED", linewidth=2.0, label="Retention")
    _annotate_promotions(ax, x, retention, rows)
    ax.set_title("False Trigger and Retention")
    ax.set_ylabel("Rate")
    ax.set_ylim(0.0, 1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=18, ha="right")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def _annotate_promotions(ax, x: list[int], y: list[float], rows: list[dict[str, str]]) -> None:
    for index, row in enumerate(rows):
        skill_id = row.get("promoted_skill_id", "")
        if not skill_id:
            continue
        stage_id = row.get("environment_stage_id", "")
        ax.annotate(
            f"{stage_id}\n{skill_id}",
            xy=(x[index], y[index]),
            xytext=(0, 18),
            textcoords="offset points",
            ha="center",
            fontsize=7,
            arrowprops={"arrowstyle": "-", "color": "#64748B", "lw": 0.8},
        )


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _as_float(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0
