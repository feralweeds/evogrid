"""Group comparison utilities."""

from __future__ import annotations

from statistics import mean, pstdev

from evogrid.evaluation.metrics_schema import METRIC_COLUMNS


def summarize_group(rows: list[dict], keys: list[str] | None = None) -> dict:
    if not rows:
        return {
            "episode_count": 0,
            "seeds": [],
            "metrics": {},
            "artifacts": {},
        }
    keys = keys or METRIC_COLUMNS
    summary = {
        "episode_count": len(rows),
        "seeds": sorted({int(row.get("seed", 0)) for row in rows}),
        "metrics": {},
        "artifacts": _artifact_summary(rows),
    }
    for key in keys:
        values = [float(row.get(key, 0.0) or 0.0) for row in rows]
        summary["metrics"][key] = {
            "mean": mean(values),
            "std": pstdev(values) if len(values) > 1 else 0.0,
            "min": min(values),
            "max": max(values),
        }
    return summary


def _artifact_summary(rows: list[dict]) -> dict:
    artifacts = {}
    for key in ["model_path", "train_log_dir", "train_config_path", "llm_trace_path"]:
        values = sorted({str(row.get(key, "")) for row in rows if row.get(key)})
        if values:
            artifacts[key + "s"] = values
    return artifacts
