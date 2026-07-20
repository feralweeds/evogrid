from __future__ import annotations

try:
    import _bootstrap  # type: ignore  # noqa: F401
except ModuleNotFoundError:
    from scripts import _bootstrap  # noqa: F401

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import yaml

from evogrid.curriculum.adaptive_controller import AdaptiveCurriculumController
from evogrid.evaluation.partitions import make_seed_partitions


CURRICULUM_GROUPS = [
    "fixed_stage",
    "random_parameter",
    "adaptive",
    "fast_promotion",
    "static_single_env",
]


def run_curriculum_ablation(config_path: str | Path, out_dir: str | Path) -> dict[str, Any]:
    config_path = Path(config_path)
    out_dir = Path(out_dir)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    experiment = dict(config.get("experiment", {}))
    root_seed = int(experiment.get("root_seed", 20260719))
    groups = [str(item) for item in experiment.get("groups", CURRICULUM_GROUPS)]
    out_dir.mkdir(parents=True, exist_ok=True)
    capability_dir = out_dir / "capability"
    capability_dir.mkdir(parents=True, exist_ok=True)

    partitions = make_seed_partitions(root_seed, {"train_map": 4, "gate_map": 4, "test_map": 6, "bootstrap": 8})
    test_seeds = partitions.partition("test").map_seeds
    rows = []
    for group_index, group in enumerate(groups):
        rows.extend(_group_rows(group, group_index, test_seeds))

    _write_csv(capability_dir / "curriculum_ablation.csv", rows)
    summary = _summarize(rows, groups)
    (capability_dir / "curriculum_ablation_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    adaptive_events = _write_adaptive_events(config, out_dir)
    manifest = {
        "schema_version": 1,
        "run_id": out_dir.name,
        "experiment_type": "curriculum_ablation",
        "mode": str(experiment.get("mode", "mock_smoke")),
        "mock_smoke": str(experiment.get("mode", "mock_smoke")) == "mock_smoke",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "config": str(config_path),
        "groups": groups,
        "seed_partitions": partitions.to_manifest(),
        "independent_test_partition": "test",
        "adaptive_events": str(adaptive_events.relative_to(out_dir)),
        "summary": summary,
        "completion_status": "completed",
        "conclusion_strength": "smoke_only_no_scientific_claim",
    }
    (out_dir / "run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return manifest


def _group_rows(group: str, group_index: int, test_seeds: list[int]) -> list[dict[str, Any]]:
    base = {
        "fixed_stage": (0.48, 0.55, 0.50),
        "random_parameter": (0.42, 0.45, 0.43),
        "adaptive": (0.61, 0.68, 0.64),
        "fast_promotion": (0.54, 0.42, 0.46),
        "static_single_env": (0.35, 0.40, 0.37),
    }.get(group, (0.4, 0.4, 0.4))
    rows = []
    for seed_index, seed in enumerate(test_seeds):
        jitter = ((seed % 11) - 5) * 0.004
        rows.append(
            {
                "schema_version": 1,
                "group": group,
                "test_seed": seed,
                "checkpoint": seed_index,
                "skill_acquisition_rate": round(_clamp(base[0] + jitter), 6),
                "retention_score": round(_clamp(base[1] + jitter / 2), 6),
                "independent_test_score": round(_clamp(base[2] + jitter + group_index * 0.001), 6),
            }
        )
    return rows


def _write_adaptive_events(config: dict[str, Any], out_dir: Path) -> Path:
    controller = AdaptiveCurriculumController.from_dict(config.get("adaptive_controller", {}))
    params = {"p_open": [0.72], "topology_hurst": [0.4], "terrain_hurst": [0.35]}
    for index, gate_score in enumerate([0.82, 0.62, 0.31]):
        event = controller.decide(
            "adaptive_stage",
            params,
            {
                "train_score": min(1.0, gate_score + 0.08),
                "gate_score": gate_score,
                "observed_partitions": ["train", "gate"],
                "window_id": f"gate_window_{index}",
                "sample_size": 4,
            },
            decision_index=index,
        )
        params = event["to_parameters"]
    return controller.write_events_jsonl(out_dir / "curriculum_events.jsonl")


def _summarize(rows: list[dict[str, Any]], groups: list[str]) -> dict[str, Any]:
    metrics = ["skill_acquisition_rate", "retention_score", "independent_test_score"]
    summary: dict[str, Any] = {"schema_version": 1, "by_group": {}}
    for group in groups:
        group_rows = [row for row in rows if row["group"] == group]
        summary["by_group"][group] = {}
        for metric in metrics:
            values = [float(row[metric]) for row in group_rows]
            summary["by_group"][group][metric] = {
                "mean": sum(values) / len(values) if values else 0.0,
                "n": len(values),
            }
    adaptive = summary["by_group"].get("adaptive", {})
    fixed = summary["by_group"].get("fixed_stage", {})
    summary["adaptive_over_fixed_test_delta"] = (
        adaptive.get("independent_test_score", {}).get("mean", 0.0)
        - fixed.get("independent_test_score", {}).get("mean", 0.0)
    )
    summary["claim_guardrail"] = "mock smoke validates plumbing only; real conclusion requires non-mock reruns"
    return summary


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _clamp(value: float) -> float:
    return min(1.0, max(0.0, value))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run M6 curriculum-ablation smoke comparison.")
    parser.add_argument("--config", default="configs/curriculum_ablation.yaml")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    manifest = run_curriculum_ablation(args.config, args.out)
    print(json.dumps({"run_manifest": str(Path(args.out) / "run_manifest.json"), "status": manifest["completion_status"]}))


if __name__ == "__main__":
    main()
