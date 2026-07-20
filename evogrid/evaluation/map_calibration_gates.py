"""Formal gate checks for map-calibration diagnostics."""

from __future__ import annotations

import csv
from dataclasses import dataclass
import json
from pathlib import Path
from statistics import median
from typing import Any


CELL_KEYS = ("height", "width", "topology_hurst", "terrain_hurst", "p_open", "resource_distribution")


@dataclass(frozen=True)
class CalibrationGate:
    gate_id: str
    name: str
    passed: bool
    details: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "gate_id": self.gate_id,
            "name": self.name,
            "passed": self.passed,
            "details": self.details,
        }


@dataclass(frozen=True)
class MapCalibrationGateReport:
    schema_version: int
    min_seeds_per_cell: int
    gates: list[CalibrationGate]

    @property
    def passed(self) -> bool:
        return all(gate.passed for gate in self.gates)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "protocol_id": "map_calibration_A0_A5_v1",
            "min_seeds_per_cell": self.min_seeds_per_cell,
            "passed": self.passed,
            "conclusion_level_if_passed": "E1",
            "gates": [gate.to_dict() for gate in self.gates],
        }


def evaluate_map_calibration_gates(
    records: list[dict[str, Any]],
    *,
    min_seeds_per_cell: int = 100,
    p_tolerance_cells: float = 1.0,
    anisotropy_tolerance: float = 0.2,
    roughness_mean_tolerance: float = 0.02,
    expected_cells: list[dict[str, Any]] | None = None,
    reproducibility_check: dict[str, Any] | None = None,
) -> MapCalibrationGateReport:
    reproducibility_check = reproducibility_check or {"checked": False}
    gates = [
        _gate_a0(records, min_seeds_per_cell, expected_cells),
        _gate_a1(records, min_seeds_per_cell, reproducibility_check),
        _gate_a2(records, p_tolerance_cells),
        _gate_a3(records, roughness_mean_tolerance),
        _gate_a4(records, anisotropy_tolerance),
        _gate_a5(records),
    ]
    return MapCalibrationGateReport(schema_version=1, min_seeds_per_cell=int(min_seeds_per_cell), gates=gates)


def evaluate_map_calibration_csv(
    diagnostics_csv: str | Path,
    *,
    min_seeds_per_cell: int = 100,
    reproducibility_check: dict[str, Any] | None = None,
) -> MapCalibrationGateReport:
    with Path(diagnostics_csv).open("r", newline="", encoding="utf-8") as handle:
        records = list(csv.DictReader(handle))
    return evaluate_map_calibration_gates(
        records,
        min_seeds_per_cell=min_seeds_per_cell,
        reproducibility_check=reproducibility_check,
    )


def write_map_calibration_gate_report(report: MapCalibrationGateReport, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report.to_dict(), ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    return path


def _gate_a0(
    records: list[dict[str, Any]],
    min_seeds_per_cell: int,
    expected_cells: list[dict[str, Any]] | None,
) -> CalibrationGate:
    groups = _group_by_cell(records)
    counts = [len(rows) for rows in groups.values()]
    no_nan_inf = all(_row_has_no_nan_inf(row) for row in records)
    map_ids = [str(row.get("map_id", "")) for row in records]
    unique_map_ids = len(map_ids) == len(set(map_ids))
    duplicate_cell_seed_count = _duplicate_cell_seed_count(records)
    placement_ok = all(str(row.get("placement_status", "ok")) == "ok" for row in records)
    percolation_valid = all(_bool(row.get("valid_for_percolation_analysis", True)) for row in records)
    min_count = min(counts) if counts else 0
    expected_cell_keys = {_canonical_cell(cell) for cell in (expected_cells or [])}
    observed_cell_keys = set(groups)
    missing_cells = expected_cell_keys - observed_cell_keys
    unexpected_cells = observed_cell_keys - expected_cell_keys if expected_cell_keys else set()
    expected_cells_match = not expected_cell_keys or (not missing_cells and not unexpected_cells)
    return CalibrationGate(
        "A0",
        "completeness",
        bool(records)
        and min_count >= min_seeds_per_cell
        and no_nan_inf
        and unique_map_ids
        and duplicate_cell_seed_count == 0
        and placement_ok
        and percolation_valid
        and expected_cells_match,
        {
            "cell_count": len(groups),
            "expected_cell_count": len(expected_cell_keys) if expected_cell_keys else None,
            "missing_cell_count": len(missing_cells),
            "unexpected_cell_count": len(unexpected_cells),
            "min_observed_seeds_per_cell": min_count,
            "required_seeds_per_cell": int(min_seeds_per_cell),
            "no_nan_inf": no_nan_inf,
            "unique_map_ids": unique_map_ids,
            "duplicate_cell_seed_count": duplicate_cell_seed_count,
            "placement_ok": placement_ok,
            "valid_for_percolation_analysis": percolation_valid,
        },
    )


def _gate_a1(
    records: list[dict[str, Any]],
    min_seeds_per_cell: int,
    reproducibility_check: dict[str, Any],
) -> CalibrationGate:
    duplicate_conflicts = []
    seen: dict[tuple[Any, ...], str] = {}
    for row in records:
        key = tuple(row.get(key) for key in (*CELL_KEYS, "seed"))
        map_id = str(row.get("map_id", ""))
        if key in seen and seen[key] != map_id:
            duplicate_conflicts.append({"key": key, "first": seen[key], "second": map_id})
        seen[key] = map_id
    checked = bool(reproducibility_check.get("checked", False))
    failures = int(reproducibility_check.get("failure_count", 0) or 0)
    checked_count = int(reproducibility_check.get("checked_count", 0) or 0)
    required_checked_count = int(reproducibility_check.get("required_checked_count", min(30, len(records), min_seeds_per_cell)))
    return CalibrationGate(
        "A1",
        "reproducibility",
        checked and checked_count >= required_checked_count and failures == 0 and not duplicate_conflicts,
        {
            "rebuild_hash_check_present": checked,
            "required_rebuild_hash_checked_count": required_checked_count,
            "rebuild_hash_checked_count": checked_count,
            "rebuild_hash_failure_count": failures,
            "duplicate_conflict_count": len(duplicate_conflicts),
        },
    )


def _gate_a2(records: list[dict[str, Any]], p_tolerance_cells: float) -> CalibrationGate:
    error_rows = []
    for row in records:
        target = _float(row.get("target_p_open", row.get("p_open")))
        realized = _float(row.get("realized_p_open"))
        height = max(1, int(_float(row.get("height"), 1)))
        width = max(1, int(_float(row.get("width"), 1)))
        tolerance = float(p_tolerance_cells) / float(height * width)
        if abs(realized - target) > tolerance + 1e-12:
            error_rows.append(abs(realized - target))
    span_report = _span_probability_monotonic_report(records)
    monotonic = span_report["hard_violation_count"] == 0
    return CalibrationGate(
        "A2",
        "p_open_control",
        not error_rows and monotonic,
        {
            "max_realized_error": max(error_rows) if error_rows else 0.0,
            "span_probability_non_decreasing": monotonic,
            **span_report,
        },
    )


def _gate_a3(records: list[dict[str, Any]], roughness_mean_tolerance: float) -> CalibrationGate:
    terrain_groups: dict[float, list[float]] = {}
    topology_groups: dict[float, list[float]] = {}
    terrain_neighbor_groups: dict[float, list[float]] = {}
    topology_neighbor_groups: dict[float, list[float]] = {}
    roughness_mean_groups: dict[float, list[float]] = {}
    for row in records:
        if row.get("estimated_terrain_hurst") not in {None, ""}:
            terrain_groups.setdefault(_float(row.get("terrain_hurst")), []).append(_float(row.get("estimated_terrain_hurst")))
        if row.get("estimated_topology_hurst") not in {None, ""}:
            topology_groups.setdefault(_float(row.get("topology_hurst")), []).append(_float(row.get("estimated_topology_hurst")))
        if row.get("terrain_neighbor_correlation") not in {None, ""}:
            terrain_neighbor_groups.setdefault(_float(row.get("terrain_hurst")), []).append(
                _float(row.get("terrain_neighbor_correlation"))
            )
        if row.get("topology_neighbor_correlation") not in {None, ""}:
            topology_neighbor_groups.setdefault(_float(row.get("topology_hurst")), []).append(
                _float(row.get("topology_neighbor_correlation"))
            )
        if row.get("roughness_mean") not in {None, ""}:
            roughness_mean_groups.setdefault(_float(row.get("terrain_hurst")), []).append(_float(row.get("roughness_mean")))
    terrain_ordered = _median_ordered(terrain_groups)
    topology_ordered = _median_ordered(topology_groups)
    terrain_neighbor_ordered = _median_ordered(terrain_neighbor_groups)
    topology_neighbor_ordered = _median_ordered(topology_neighbor_groups)
    roughness_medians = _group_medians(roughness_mean_groups)
    roughness_mean_range = max(roughness_medians.values()) - min(roughness_medians.values()) if roughness_medians else None
    roughness_marginal_stable = roughness_mean_range is not None and roughness_mean_range <= roughness_mean_tolerance
    terrain_spatial_metric_ordered = terrain_ordered or terrain_neighbor_ordered
    topology_spatial_metric_ordered = topology_ordered or topology_neighbor_ordered
    terrain_applicable = len(terrain_groups) >= 3 or len(terrain_neighbor_groups) >= 3
    topology_applicable = len(topology_groups) >= 3 or len(topology_neighbor_groups) >= 3
    terrain_passed = not terrain_applicable or terrain_spatial_metric_ordered
    topology_passed = not topology_applicable or topology_spatial_metric_ordered
    at_least_one_h_axis_scanned = terrain_applicable or topology_applicable
    return CalibrationGate(
        "A3",
        "hurst_separability",
        at_least_one_h_axis_scanned
        and terrain_passed
        and topology_passed
        and roughness_marginal_stable,
        {
            "terrain_h_levels": sorted(terrain_groups),
            "topology_h_levels": sorted(topology_groups),
            "terrain_axis_applicable": terrain_applicable,
            "topology_axis_applicable": topology_applicable,
            "terrain_axis_passed": terrain_passed,
            "topology_axis_passed": topology_passed,
            "terrain_medians_ordered": terrain_ordered,
            "topology_medians_ordered": topology_ordered,
            "terrain_neighbor_correlation_medians_ordered": terrain_neighbor_ordered,
            "topology_neighbor_correlation_medians_ordered": topology_neighbor_ordered,
            "terrain_neighbor_correlation_medians": _group_medians(terrain_neighbor_groups),
            "topology_neighbor_correlation_medians": _group_medians(topology_neighbor_groups),
            "roughness_mean_medians": roughness_medians,
            "roughness_mean_range": roughness_mean_range,
            "roughness_mean_tolerance": float(roughness_mean_tolerance),
            "roughness_marginal_stable": roughness_marginal_stable,
            "pre_registered_spatial_metric_ordered": terrain_spatial_metric_ordered and topology_spatial_metric_ordered,
        },
    )


def _gate_a4(records: list[dict[str, Any]], anisotropy_tolerance: float) -> CalibrationGate:
    cell_reports = []
    for cell, rows in _group_by_cell(records).items():
        horizontal_values = [_bool(row.get("spans_horizontal")) for row in rows]
        vertical_values = [_bool(row.get("spans_vertical")) for row in rows]
        horizontal = _binomial_summary(horizontal_values)
        vertical = _binomial_summary(vertical_values)
        diff = abs(horizontal["rate"] - vertical["rate"])
        ci_separated = horizontal["ci_low"] > vertical["ci_high"] or vertical["ci_low"] > horizontal["ci_high"]
        terrain_corr_diff = _median_optional(row.get("terrain_axis_corr_abs_diff") for row in rows)
        topology_corr_diff = _median_optional(row.get("topology_axis_corr_abs_diff") for row in rows)
        terrain_semivar_rel_diff = _median_relative_axis_diff(
            rows,
            "terrain_lag1_semivariance_horizontal",
            "terrain_lag1_semivariance_vertical",
        )
        topology_semivar_rel_diff = _median_relative_axis_diff(
            rows,
            "topology_lag1_semivariance_horizontal",
            "topology_lag1_semivariance_vertical",
        )
        cell_reports.append(
            {
                "cell": _cell_dict(cell),
                "sample_size": len(rows),
                "horizontal": horizontal,
                "vertical": vertical,
                "abs_rate_diff": diff,
                "ci_separated": ci_separated,
                "terrain_axis_corr_abs_diff_median": terrain_corr_diff,
                "topology_axis_corr_abs_diff_median": topology_corr_diff,
                "terrain_lag1_semivariance_relative_diff_median": terrain_semivar_rel_diff,
                "topology_lag1_semivariance_relative_diff_median": topology_semivar_rel_diff,
            }
        )
    diffs = [item["abs_rate_diff"] for item in cell_reports]
    separated = [item for item in cell_reports if item["ci_separated"] and item["abs_rate_diff"] > anisotropy_tolerance]
    max_diff = max(diffs) if diffs else 1.0
    corr_diffs = [
        value
        for item in cell_reports
        for value in (item["terrain_axis_corr_abs_diff_median"], item["topology_axis_corr_abs_diff_median"])
        if value is not None
    ]
    semivar_diffs = [
        value
        for item in cell_reports
        for value in (
            item["terrain_lag1_semivariance_relative_diff_median"],
            item["topology_lag1_semivariance_relative_diff_median"],
        )
        if value is not None
    ]
    max_axis_corr_diff = max(corr_diffs) if corr_diffs else None
    max_axis_semivariance_relative_diff = max(semivar_diffs) if semivar_diffs else None
    axis_corr_ok = max_axis_corr_diff is not None and max_axis_corr_diff <= anisotropy_tolerance
    axis_semivar_ok = (
        max_axis_semivariance_relative_diff is not None
        and max_axis_semivariance_relative_diff <= anisotropy_tolerance
    )
    return CalibrationGate(
        "A4",
        "axis_anisotropy",
        bool(diffs) and not separated and axis_corr_ok and axis_semivar_ok,
        {
            "max_span_probability_axis_diff": max_diff,
            "max_axis_correlation_abs_diff": max_axis_corr_diff,
            "max_axis_semivariance_relative_diff": max_axis_semivariance_relative_diff,
            "tolerance": float(anisotropy_tolerance),
            "significant_axis_bias_count": len(separated),
            "axis_correlation_within_tolerance": axis_corr_ok,
            "axis_semivariance_within_tolerance": axis_semivar_ok,
            "cell_reports": cell_reports,
        },
    )


def _gate_a5(records: list[dict[str, Any]]) -> CalibrationGate:
    reports = _critical_curve_reports(records)
    estimable = [item for item in reports if item["p50_estimable"]]
    return CalibrationGate(
        "A5",
        "critical_curve_estimation",
        bool(reports) and len(estimable) == len(reports),
        {
            "group_count": len(reports),
            "estimable_group_count": len(estimable),
            "p50_interpolation_method": "linear_between_adjacent_p_span_points",
            "critical_curve_reports": reports,
        },
    )


def _group_by_cell(records: list[dict[str, Any]]) -> dict[tuple[Any, ...], list[dict[str, Any]]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in records:
        groups.setdefault(tuple(row.get(key) for key in CELL_KEYS), []).append(row)
    return groups


def _canonical_cell(row: dict[str, Any]) -> tuple[Any, ...]:
    return tuple(_canonical_cell_value(row.get(key)) for key in CELL_KEYS)


def _canonical_cell_value(value: Any) -> Any:
    try:
        return float(value)
    except (TypeError, ValueError):
        return str(value)


def _duplicate_cell_seed_count(records: list[dict[str, Any]]) -> int:
    keys = [(*_canonical_cell(row), int(_float(row.get("seed")))) for row in records]
    return len(keys) - len(set(keys))


def _row_has_no_nan_inf(row: dict[str, Any]) -> bool:
    for value in row.values():
        if value in {None, ""}:
            continue
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if number != number or number in {float("inf"), float("-inf")}:
            return False
    return True


def _span_probability_monotonic_report(records: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[tuple[Any, ...], dict[float, list[bool]]] = {}
    for row in records:
        key = (
            row.get("height"),
            row.get("width"),
            row.get("topology_hurst"),
            row.get("terrain_hurst"),
            row.get("resource_distribution"),
        )
        grouped.setdefault(key, {}).setdefault(_float(row.get("p_open")), []).append(
            _bool(row.get("spans_horizontal")) or _bool(row.get("spans_vertical"))
        )
    group_reports = []
    hard_violations = []
    uncertain_nonmonotonic = []
    for key, p_groups in sorted(grouped.items(), key=lambda item: str(item[0])):
        points = []
        for p_open in sorted(p_groups):
            summary = _binomial_summary(p_groups[p_open])
            points.append({"p_open": p_open, **summary})
        for left, right in zip(points, points[1:]):
            if right["rate"] + 1e-12 < left["rate"]:
                item = {
                    "cell_group": _span_group_dict(key),
                    "left": left,
                    "right": right,
                    "ci_overlaps": right["ci_high"] >= left["ci_low"],
                }
                if item["ci_overlaps"]:
                    uncertain_nonmonotonic.append(item)
                else:
                    hard_violations.append(item)
        group_reports.append({"cell_group": _span_group_dict(key), "points": points})
    return {
        "span_probability_groups": group_reports,
        "hard_violation_count": len(hard_violations),
        "uncertain_nonmonotonic_count": len(uncertain_nonmonotonic),
        "hard_violations": hard_violations,
        "uncertain_nonmonotonic": uncertain_nonmonotonic,
    }


def _critical_curve_reports(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], dict[float, list[bool]]] = {}
    for row in records:
        key = (
            row.get("height"),
            row.get("width"),
            row.get("topology_hurst"),
            row.get("terrain_hurst"),
            row.get("resource_distribution"),
        )
        grouped.setdefault(key, {}).setdefault(_float(row.get("p_open")), []).append(
            _bool(row.get("spans_horizontal")) or _bool(row.get("spans_vertical"))
        )
    reports = []
    for key, p_groups in sorted(grouped.items(), key=lambda item: str(item[0])):
        points = []
        for p_open in sorted(p_groups):
            points.append({"p_open": p_open, **_binomial_summary(p_groups[p_open])})
        p50 = _estimate_p50(points)
        reports.append(
            {
                "cell_group": _span_group_dict(key),
                "points": points,
                "p50_estimable": p50 is not None,
                "p50": p50,
                "finite_size_only": True,
            }
        )
    return reports


def _estimate_p50(points: list[dict[str, Any]]) -> float | None:
    if not points:
        return None
    for point in points:
        if abs(float(point["rate"]) - 0.5) <= 1e-12:
            return float(point["p_open"])
    for left, right in zip(points, points[1:]):
        left_rate = float(left["rate"])
        right_rate = float(right["rate"])
        if (left_rate - 0.5) * (right_rate - 0.5) <= 0.0 and left_rate != right_rate:
            t = (0.5 - left_rate) / (right_rate - left_rate)
            return float(left["p_open"]) + t * (float(right["p_open"]) - float(left["p_open"]))
    return None


def _median_ordered(groups: dict[float, list[float]]) -> bool:
    if len(groups) < 2:
        return False
    medians = [median(groups[key]) for key in sorted(groups)]
    return all(curr > prev for prev, curr in zip(medians, medians[1:]))


def _group_medians(groups: dict[float, list[float]]) -> dict[float, float]:
    return {key: median(values) for key, values in sorted(groups.items()) if values}


def _median_optional(values: Any) -> float | None:
    clean = []
    for value in values:
        if value in {None, ""}:
            continue
        clean.append(_float(value))
    return None if not clean else median(clean)


def _median_relative_axis_diff(
    rows: list[dict[str, Any]],
    horizontal_key: str,
    vertical_key: str,
) -> float | None:
    values = []
    for row in rows:
        if row.get(horizontal_key) in {None, ""} or row.get(vertical_key) in {None, ""}:
            continue
        horizontal = _float(row.get(horizontal_key))
        vertical = _float(row.get(vertical_key))
        scale = (abs(horizontal) + abs(vertical)) / 2.0
        if scale == 0.0:
            values.append(0.0)
        else:
            values.append(abs(horizontal - vertical) / scale)
    return None if not values else median(values)


def _mean(values: list[bool]) -> float:
    return sum(1.0 for value in values if value) / len(values) if values else 0.0


def _binomial_summary(values: list[bool], z: float = 1.96) -> dict[str, Any]:
    n = len(values)
    successes = sum(1 for value in values if value)
    if n == 0:
        return {"successes": 0, "n": 0, "rate": 0.0, "ci_low": 0.0, "ci_high": 1.0, "ci_method": "wilson"}
    p_hat = successes / n
    denom = 1.0 + z * z / n
    center = (p_hat + z * z / (2.0 * n)) / denom
    margin = z * ((p_hat * (1.0 - p_hat) / n + z * z / (4.0 * n * n)) ** 0.5) / denom
    return {
        "successes": successes,
        "n": n,
        "rate": p_hat,
        "ci_low": max(0.0, center - margin),
        "ci_high": min(1.0, center + margin),
        "ci_method": "wilson",
    }


def _cell_dict(cell: tuple[Any, ...]) -> dict[str, Any]:
    return {key: cell[index] for index, key in enumerate(CELL_KEYS)}


def _span_group_dict(group: tuple[Any, ...]) -> dict[str, Any]:
    keys = ("height", "width", "topology_hurst", "terrain_hurst", "resource_distribution")
    return {key: group[index] for index, key in enumerate(keys)}


def _float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes"}
