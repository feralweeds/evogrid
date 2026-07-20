from __future__ import annotations

try:
    import _bootstrap  # type: ignore  # noqa: F401
except ModuleNotFoundError:
    from scripts import _bootstrap  # noqa: F401

import argparse
import csv
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
from importlib import metadata
import json
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import yaml

from evogrid.constants import Tile
from evogrid.evaluation.map_calibration_gates import (
    evaluate_map_calibration_gates,
    write_map_calibration_gate_report,
)
from evogrid.envs.map_builder import build_map


def run_calibration(config_path: str | Path, out_dir: str | Path, seeds_spec: str) -> dict[str, Any]:
    config_path = Path(config_path)
    out_dir = Path(out_dir)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    seeds = _parse_seeds(seeds_spec)
    planned_cells = _iter_parameter_grid(config)
    expected_cells = [_cell_key(params) for params in planned_cells]
    representative_targets = _representative_targets(planned_cells, seeds)
    resolved_hash = _stable_hash(config)
    started_at = datetime.now(timezone.utc).isoformat()

    maps_dir = out_dir / "maps"
    figures_dir = out_dir / "figures"
    maps_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config_resolved.yaml").write_text(
        yaml.safe_dump(config, sort_keys=True, allow_unicode=True),
        encoding="utf-8",
    )

    records: list[dict[str, Any]] = []
    representative_records: list[dict[str, Any]] = []
    map_manifest_path = maps_dir / "map_manifest.jsonl"
    with map_manifest_path.open("w", encoding="utf-8") as manifest_file:
        for params in planned_cells:
            for seed in seeds:
                run_config = _apply_params(config, params)
                result = build_map(run_config, seed=seed)
                record = {
                    "schema_version": 1,
                    "seed": seed,
                    **params,
                    "map_id": result.map_id,
                    **_flat_diagnostics(result.diagnostics),
                }
                records.append(record)
                if _representative_key(params, seed) in representative_targets:
                    representative_records.append({**record, "_grid": result.grid})
                manifest_file.write(
                    json.dumps(
                        {
                            "schema_version": 1,
                            "seed": seed,
                            "params": params,
                            "map_id": result.map_id,
                            "diagnostics": result.diagnostics,
                        },
                        ensure_ascii=False,
                        sort_keys=True,
                    )
                    + "\n"
                )

    diagnostics_path = maps_dir / "diagnostics.csv"
    _write_csv(diagnostics_path, records)
    calibration = config.get("calibration", {})
    reproducibility_check = _rebuild_reproducibility_check(
        config,
        records,
        sample_count=int(calibration.get("reproducibility_sample_count", 0)),
    )
    gate_report = evaluate_map_calibration_gates(
        records,
        min_seeds_per_cell=int(calibration.get("formal_min_seeds_per_cell", 100)),
        expected_cells=expected_cells,
        reproducibility_check=reproducibility_check,
    )
    gate_report_path = write_map_calibration_gate_report(gate_report, maps_dir / "calibration_gates.json")
    curve_outputs = _write_curve_outputs(maps_dir, gate_report.to_dict())
    figure_paths = _write_figures(figures_dir, records, curve_outputs, representative_records)
    manifest = {
        "schema_version": 1,
        "run_id": out_dir.name,
        "experiment_type": config.get("calibration", {}).get("experiment_type", "map_calibration_smoke"),
        "mode": config.get("calibration", {}).get("mode", _infer_mode(config)),
        "mock_smoke": False,
        "started_at": started_at,
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "config_path": str(config_path),
        "resolved_config_hash": resolved_hash,
        "seed_spec": seeds_spec,
        "seeds": seeds,
        "seed_count": len(seeds),
        "planned_cell_count": len(planned_cells),
        "expected_map_count": len(planned_cells) * len(seeds),
        "map_count": len(records),
        "integrity": _calibration_integrity(records, expected_cells, seeds),
        "runtime": _runtime_metadata(),
        "outputs": {
            "config_resolved": "config_resolved.yaml",
            "calibration_gates": "maps/calibration_gates.json",
            "diagnostics_csv": "maps/diagnostics.csv",
            "p_span_curves": "maps/p_span_curves.csv",
            "p_span_curves_json": "maps/p_span_curves.json",
            "p50_summary": "maps/p50_summary.csv",
            "p50_summary_json": "maps/p50_summary.json",
            "map_manifest": "maps/map_manifest.jsonl",
            "figures": [str(path.relative_to(out_dir)).replace("\\", "/") for path in figure_paths],
        },
        "reproducibility_check": reproducibility_check,
        "formal_acceptance": {
            "passed": gate_report.passed,
            "conclusion_level": "E1" if gate_report.passed else "E0",
            "gate_report": str(gate_report_path.relative_to(out_dir)),
        },
        "completion_status": "completed",
    }
    (out_dir / "run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate fractal/percolation map statistics.")
    parser.add_argument("--config", default="configs/map_calibration.yaml")
    parser.add_argument("--out", required=True)
    parser.add_argument("--seeds", default="0:5")
    args = parser.parse_args()

    manifest = run_calibration(args.config, args.out, args.seeds)
    print(json.dumps({"run_manifest": str(Path(args.out) / "run_manifest.json"), "map_count": manifest["map_count"]}))


def _iter_parameter_grid(config: dict[str, Any]) -> list[dict[str, Any]]:
    calibration = config.get("calibration", {})
    sizes = calibration.get("sizes", [config.get("env", {}).get("grid_size", [64, 64])])
    topology_hurst = calibration.get("topology_hurst", [config["env"]["world"]["topology"].get("hurst", 0.5)])
    terrain_hurst = calibration.get("terrain_hurst", [config["env"]["world"]["terrain"].get("hurst", 0.7)])
    p_values = calibration.get("p_open", [config["env"]["world"]["topology"].get("p_open", 0.65)])
    resource_distributions = calibration.get(
        "resource_distribution",
        [config["env"]["world"]["resources"].get("distribution", "uniform")],
    )
    params = []
    for size in sizes:
        for topology_h in topology_hurst:
            for terrain_h in terrain_hurst:
                for p_open in p_values:
                    for distribution in resource_distributions:
                        params.append(
                            {
                                "height": int(size[0]),
                                "width": int(size[1]),
                                "topology_hurst": float(topology_h),
                                "terrain_hurst": float(terrain_h),
                                "p_open": float(p_open),
                                "resource_distribution": str(distribution),
                            }
                        )
    return params


def _cell_key(params: dict[str, Any]) -> dict[str, Any]:
    return {
        "height": int(params["height"]),
        "width": int(params["width"]),
        "topology_hurst": float(params["topology_hurst"]),
        "terrain_hurst": float(params["terrain_hurst"]),
        "p_open": float(params["p_open"]),
        "resource_distribution": str(params["resource_distribution"]),
    }


def _representative_targets(planned_cells: list[dict[str, Any]], seeds: list[int]) -> set[tuple[Any, ...]]:
    if not planned_cells or not seeds:
        return set()
    min_size = min((int(cell["height"]) * int(cell["width"]), int(cell["height"]), int(cell["width"])) for cell in planned_cells)
    smallest_cells = [
        cell for cell in planned_cells if (int(cell["height"]) * int(cell["width"]), int(cell["height"]), int(cell["width"])) == min_size
    ]
    p_values = sorted({float(cell["p_open"]) for cell in smallest_cells})
    representative_p = p_values[len(p_values) // 2]
    seed = seeds[0]
    targets = set()
    seen_h: set[float] = set()
    for cell in sorted(smallest_cells, key=lambda row: (float(row["topology_hurst"]), float(row["terrain_hurst"]))):
        topology_h = float(cell["topology_hurst"])
        if topology_h in seen_h or float(cell["p_open"]) != representative_p:
            continue
        seen_h.add(topology_h)
        targets.add(_representative_key(cell, seed))
    return targets


def _representative_key(params: dict[str, Any], seed: int) -> tuple[Any, ...]:
    return (
        int(params["height"]),
        int(params["width"]),
        float(params["topology_hurst"]),
        float(params["terrain_hurst"]),
        float(params["p_open"]),
        str(params["resource_distribution"]),
        int(seed),
    )


def _apply_params(config: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
    run_config = deepcopy(config)
    env = run_config["env"]
    env["grid_size"] = [params["height"], params["width"]]
    env["world"]["topology"]["hurst"] = params["topology_hurst"]
    env["world"]["topology"]["p_open"] = params["p_open"]
    env["world"]["terrain"]["hurst"] = params["terrain_hurst"]
    env["world"]["resources"]["distribution"] = params["resource_distribution"]
    return run_config


def _write_csv(path: Path, records: list[dict[str, Any]]) -> None:
    if not records:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({key for record in records for key in record})
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def _write_curve_outputs(maps_dir: Path, gate_report: dict[str, Any]) -> dict[str, Any]:
    a5 = next((gate for gate in gate_report.get("gates", []) if gate.get("gate_id") == "A5"), {})
    reports = a5.get("details", {}).get("critical_curve_reports", [])
    curves: list[dict[str, Any]] = []
    p50_rows: list[dict[str, Any]] = []
    for report in reports:
        group = report.get("cell_group", {})
        p50_rows.append(
            {
                "schema_version": 1,
                **group,
                "p50": report.get("p50"),
                "p50_estimable": report.get("p50_estimable"),
                "finite_size_only": report.get("finite_size_only", True),
                "p50_interpolation_method": a5.get("details", {}).get("p50_interpolation_method"),
            }
        )
        for point in report.get("points", []):
            curves.append(
                {
                    "schema_version": 1,
                    **group,
                    "p_open": point.get("p_open"),
                    "span_successes": point.get("successes"),
                    "span_n": point.get("n"),
                    "p_span": point.get("rate"),
                    "ci_low": point.get("ci_low"),
                    "ci_high": point.get("ci_high"),
                    "ci_method": point.get("ci_method"),
                }
            )

    curves_csv = maps_dir / "p_span_curves.csv"
    p50_csv = maps_dir / "p50_summary.csv"
    _write_csv(curves_csv, curves)
    _write_csv(p50_csv, p50_rows)
    curves_json = maps_dir / "p_span_curves.json"
    p50_json = maps_dir / "p50_summary.json"
    curves_json.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "protocol_id": gate_report.get("protocol_id"),
                "ci_method": "wilson",
                "curves": curves,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    p50_json.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "protocol_id": gate_report.get("protocol_id"),
                "finite_size_only": True,
                "p50_interpolation_method": a5.get("details", {}).get("p50_interpolation_method"),
                "p50": p50_rows,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return {"curves": curves, "p50": p50_rows}


def _write_figures(
    figures_dir: Path,
    records: list[dict[str, Any]],
    curve_outputs: dict[str, Any],
    representative_records: list[dict[str, Any]],
) -> list[Path]:
    if not records:
        return []
    paths = [
        _plot_p_span_curves(figures_dir, curve_outputs.get("curves", [])),
        _plot_largest_component_curves(figures_dir, records),
        _plot_h_calibration(figures_dir, records),
        _plot_rough_patch_distribution(figures_dir, records),
        _plot_articulation_stats(figures_dir, records),
        _plot_axis_anisotropy(figures_dir, records),
        _plot_representative_maps(figures_dir, representative_records),
    ]
    return [path for path in paths if path is not None]


def _plot_p_span_curves(figures_dir: Path, curves: list[dict[str, Any]]) -> Path | None:
    if not curves:
        return None
    path = figures_dir / "p_span_curves_with_ci.png"
    sizes = sorted({(int(row["height"]), int(row["width"])) for row in curves})
    fig, axes = _subplots_for_count(len(sizes), height_per_panel=3.2)
    for axis, size in zip(axes, sizes):
        size_rows = [row for row in curves if (int(row["height"]), int(row["width"])) == size]
        for topology_h in sorted({float(row["topology_hurst"]) for row in size_rows}):
            rows = sorted([row for row in size_rows if float(row["topology_hurst"]) == topology_h], key=lambda row: float(row["p_open"]))
            xs = np.array([float(row["p_open"]) for row in rows], dtype=float)
            ys = np.array([float(row["p_span"]) for row in rows], dtype=float)
            low = np.array([float(row["ci_low"]) for row in rows], dtype=float)
            high = np.array([float(row["ci_high"]) for row in rows], dtype=float)
            axis.plot(xs, ys, marker="o", label=f"H={topology_h:g}")
            axis.fill_between(xs, low, high, alpha=0.15)
        axis.set_title(f"size {size[0]}x{size[1]}")
        axis.set_xlabel("p_open")
        axis.set_ylabel("P_span")
        axis.set_ylim(-0.05, 1.05)
        axis.legend()
    _hide_unused_axes(axes, len(sizes))
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def _plot_largest_component_curves(figures_dir: Path, records: list[dict[str, Any]]) -> Path:
    path = figures_dir / "largest_component_fraction_curves.png"
    groups = _mean_by(records, ["height", "width", "topology_hurst", "p_open"], "largest_component_fraction")
    sizes = sorted({(int(row["height"]), int(row["width"])) for row in groups})
    fig, axes = _subplots_for_count(len(sizes), height_per_panel=3.2)
    for axis, size in zip(axes, sizes):
        size_rows = [row for row in groups if (int(row["height"]), int(row["width"])) == size]
        for topology_h in sorted({float(row["topology_hurst"]) for row in size_rows}):
            rows = sorted([row for row in size_rows if float(row["topology_hurst"]) == topology_h], key=lambda row: float(row["p_open"]))
            axis.plot([row["p_open"] for row in rows], [row["mean"] for row in rows], marker="o", label=f"H={topology_h:g}")
        axis.set_title(f"size {size[0]}x{size[1]}")
        axis.set_xlabel("p_open")
        axis.set_ylabel("largest component fraction")
        axis.set_ylim(-0.05, 1.05)
        axis.legend()
    _hide_unused_axes(axes, len(sizes))
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def _plot_h_calibration(figures_dir: Path, records: list[dict[str, Any]]) -> Path:
    path = figures_dir / "input_h_vs_estimated_h.png"
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    axes[0].scatter(
        [float(record["topology_hurst"]) for record in records],
        [_float_or_zero(record.get("estimated_topology_hurst")) for record in records],
        s=14,
    )
    axes[0].set_xlabel("input topology H")
    axes[0].set_ylabel("estimated topology H proxy")
    axes[1].scatter(
        [float(record["terrain_hurst"]) for record in records],
        [_float_or_zero(record.get("estimated_terrain_hurst")) for record in records],
        s=14,
    )
    axes[1].set_xlabel("input terrain H")
    axes[1].set_ylabel("estimated terrain H proxy")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def _plot_rough_patch_distribution(figures_dir: Path, records: list[dict[str, Any]]) -> Path:
    path = figures_dir / "rough_patch_size_distribution.png"
    plt.figure(figsize=(6, 4))
    for terrain_h in sorted({float(record["terrain_hurst"]) for record in records}):
        values = [
            _float_or_zero(record.get("largest_rough_patch_fraction"))
            for record in records
            if float(record["terrain_hurst"]) == terrain_h
        ]
        plt.hist(values, bins=12, alpha=0.35, label=f"terrain H={terrain_h:g}")
    plt.xlabel("largest rough patch fraction")
    plt.ylabel("map count")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    return path


def _plot_articulation_stats(figures_dir: Path, records: list[dict[str, Any]]) -> Path:
    path = figures_dir / "articulation_bottleneck_stats.png"
    groups = _mean_by(records, ["height", "width", "topology_hurst", "p_open"], "articulation_point_count")
    plt.figure(figsize=(7, 4))
    for topology_h in sorted({float(row["topology_hurst"]) for row in groups}):
        rows = sorted([row for row in groups if float(row["topology_hurst"]) == topology_h], key=lambda row: float(row["p_open"]))
        plt.plot([row["p_open"] for row in rows], [row["mean"] for row in rows], marker="o", label=f"H={topology_h:g}")
    plt.xlabel("p_open")
    plt.ylabel("mean articulation point count")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    return path


def _plot_axis_anisotropy(figures_dir: Path, records: list[dict[str, Any]]) -> Path:
    path = figures_dir / "axis_anisotropy_differences.png"
    metrics = [
        ("terrain_axis_corr_abs_diff", "terrain corr"),
        ("topology_axis_corr_abs_diff", "topology corr"),
        ("terrain_lag1_semivariance_abs_diff", "terrain semivar"),
        ("topology_lag1_semivariance_abs_diff", "topology semivar"),
    ]
    labels = [label for _, label in metrics]
    values = []
    for key, _ in metrics:
        clean = [_float_or_zero(record.get(key)) for record in records if record.get(key) not in {None, ""}]
        values.append(float(np.median(clean)) if clean else 0.0)
    plt.figure(figsize=(7, 4))
    plt.bar(labels, values)
    plt.ylabel("median horizontal/vertical absolute difference")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    return path


def _plot_representative_maps(figures_dir: Path, records: list[dict[str, Any]]) -> Path:
    path = figures_dir / "representative_maps_by_h.png"
    selected = sorted(records, key=lambda row: (float(row["topology_hurst"]), int(row["height"]), float(row["p_open"])))
    fig, axes = _subplots_for_count(len(selected), height_per_panel=3.0)
    for axis, record in zip(axes, selected):
        grid = np.asarray(record.get("_grid", []), dtype=int)
        if grid.size == 0:
            axis.text(0.5, 0.5, "missing grid", ha="center", va="center")
        else:
            open_mask = np.isin(grid, [int(Tile.GROUND), int(Tile.BASE), int(Tile.ORE), int(Tile.ROAD)])
            axis.imshow(open_mask, cmap="gray", interpolation="nearest")
        axis.set_title(
            f"size {int(record['height'])} H={float(record['topology_hurst']):g} p={float(record['p_open']):g}"
        )
        axis.set_xticks([])
        axis.set_yticks([])
    _hide_unused_axes(axes, len(selected))
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def _subplots_for_count(count: int, height_per_panel: float = 3.0) -> tuple[Any, list[Any]]:
    count = max(1, int(count))
    cols = min(3, count)
    rows = int(np.ceil(count / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.0 * cols, height_per_panel * rows))
    if not isinstance(axes, np.ndarray):
        return fig, [axes]
    return fig, list(axes.reshape(-1))


def _hide_unused_axes(axes: list[Any], used_count: int) -> None:
    for axis in axes[used_count:]:
        axis.set_visible(False)


def _mean_by(records: list[dict[str, Any]], keys: list[str], value_key: str) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[float]] = {}
    for record in records:
        if record.get(value_key) in {None, ""}:
            continue
        groups.setdefault(tuple(record[key] for key in keys), []).append(float(record[value_key]))
    rows = []
    for group_key, values in sorted(groups.items(), key=lambda item: str(item[0])):
        row = {key: group_key[index] for index, key in enumerate(keys)}
        row["mean"] = float(np.mean(values))
        row["n"] = len(values)
        rows.append(row)
    return rows


def _float_or_zero(value: Any) -> float:
    try:
        if value in {None, ""}:
            return 0.0
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _flat_diagnostics(diagnostics: dict[str, Any]) -> dict[str, Any]:
    flat: dict[str, Any] = {}
    for key, value in diagnostics.items():
        if isinstance(value, (str, int, float, bool)) or value is None:
            flat[key] = value
    return flat


def _rebuild_reproducibility_check(
    config: dict[str, Any],
    records: list[dict[str, Any]],
    sample_count: int,
) -> dict[str, Any]:
    if sample_count <= 0 or not records:
        return {"schema_version": 1, "checked": False, "checked_count": 0, "failure_count": 0, "failures": []}
    selected = _deterministic_sample(records, min(int(sample_count), len(records)))
    failures = []
    for record in selected:
        params = {key: record[key] for key in ("height", "width", "topology_hurst", "terrain_hurst", "p_open", "resource_distribution")}
        rebuilt = build_map(_apply_params(config, params), seed=int(record["seed"]))
        if rebuilt.map_id != record["map_id"]:
            failures.append(
                {
                    "seed": int(record["seed"]),
                    "expected_map_id": record["map_id"],
                    "rebuilt_map_id": rebuilt.map_id,
                    "params": params,
                }
            )
    return {
        "schema_version": 1,
        "checked": True,
        "required_checked_count": min(30, len(records)),
        "checked_count": len(selected),
        "failure_count": len(failures),
        "failures": failures,
    }


def _deterministic_sample(records: list[dict[str, Any]], sample_count: int) -> list[dict[str, Any]]:
    ordered = sorted(
        records,
        key=lambda row: (
            str(row.get("height")),
            str(row.get("width")),
            str(row.get("topology_hurst")),
            str(row.get("terrain_hurst")),
            str(row.get("p_open")),
            str(row.get("resource_distribution")),
            int(row.get("seed", 0)),
        ),
    )
    if sample_count >= len(ordered):
        return ordered
    if sample_count == 1:
        return [ordered[0]]
    step = (len(ordered) - 1) / float(sample_count - 1)
    indexes = sorted({int(round(index * step)) for index in range(sample_count)})
    while len(indexes) < sample_count:
        for candidate in range(len(ordered)):
            if candidate not in indexes:
                indexes.append(candidate)
                break
    return [ordered[index] for index in sorted(indexes[:sample_count])]


def _parse_seeds(spec: str) -> list[int]:
    if ":" in spec:
        start, stop = spec.split(":", 1)
        return list(range(int(start), int(stop)))
    return [int(part.strip()) for part in spec.split(",") if part.strip()]


def _calibration_integrity(
    records: list[dict[str, Any]],
    expected_cells: list[dict[str, Any]],
    seeds: list[int],
) -> dict[str, Any]:
    expected_keys = {_canonical_cell_key(cell) for cell in expected_cells}
    observed_keys = {_canonical_cell_key(record) for record in records}
    row_keys = [(*_canonical_cell_key(record), int(record["seed"])) for record in records]
    duplicate_row_count = len(row_keys) - len(set(row_keys))
    return {
        "schema_version": 1,
        "expected_cell_count": len(expected_keys),
        "observed_cell_count": len(observed_keys),
        "missing_cell_count": len(expected_keys - observed_keys),
        "unexpected_cell_count": len(observed_keys - expected_keys),
        "duplicate_cell_seed_count": duplicate_row_count,
        "duplicate_seed_count": len(seeds) - len(set(seeds)),
        "expected_map_count": len(expected_keys) * len(seeds),
        "observed_map_count": len(records),
    }


def _canonical_cell_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        int(row["height"]),
        int(row["width"]),
        float(row["topology_hurst"]),
        float(row["terrain_hurst"]),
        float(row["p_open"]),
        str(row["resource_distribution"]),
    )


def _runtime_metadata() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "python": sys.version.split()[0],
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "packages": _package_versions(["numpy", "matplotlib", "PyYAML"]),
        "git": _git_metadata(),
    }


def _package_versions(package_names: list[str]) -> dict[str, str]:
    versions = {}
    for package_name in package_names:
        try:
            versions[package_name] = metadata.version(package_name)
        except metadata.PackageNotFoundError:
            versions[package_name] = "not_installed"
    return versions


def _git_metadata() -> dict[str, Any]:
    commit = _git_output(["rev-parse", "HEAD"])
    branch = _git_output(["rev-parse", "--abbrev-ref", "HEAD"])
    status = _git_output(["status", "--short"])
    return {
        "commit": commit,
        "branch": branch,
        "dirty": bool(status),
        "status_short": status.splitlines() if status else [],
    }


def _git_output(args: list[str]) -> str:
    try:
        completed = subprocess.run(
            ["git", *args],
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
    except OSError:
        return ""
    if completed.returncode != 0:
        return ""
    return completed.stdout.strip()


def _infer_mode(config: dict[str, Any]) -> str:
    experiment_type = str(config.get("calibration", {}).get("experiment_type", ""))
    return "smoke" if experiment_type.endswith("_smoke") else "formal"


def _stable_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


if __name__ == "__main__":
    main()
