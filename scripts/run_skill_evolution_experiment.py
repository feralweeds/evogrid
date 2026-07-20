from __future__ import annotations

try:
    import _bootstrap  # type: ignore  # noqa: F401
except ModuleNotFoundError:
    from scripts import _bootstrap  # noqa: F401

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any

import yaml

from evogrid.evaluation.capability import CapabilityTask, compute_capability
from evogrid.evaluation.partitions import make_seed_partitions
from evogrid.llm.skill_prompts import (
    SKILL_PROPOSER_PROMPT_ID,
    SKILL_PROPOSER_PROMPT_VERSION,
    skill_proposer_prompt_hash,
)
from evogrid.skills.schemas import SkillSpec


GROUPS = [
    "no_skill",
    "prompt_only",
    "handcrafted_skill",
    "self_proposed_candidate",
    "self_proposed_verified",
    "prompt_skill",
]


def run_skill_evolution_experiment(config_path: str | Path, out_dir: str | Path) -> dict[str, Any]:
    config_path = Path(config_path)
    out_dir = Path(out_dir)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) if config_path.exists() else {}
    config_hash = _sha256(config)
    existing_manifest = out_dir / "run_manifest.json"
    if existing_manifest.exists():
        existing = json.loads(existing_manifest.read_text(encoding="utf-8"))
        if existing.get("config_hash") != config_hash:
            raise ValueError("resume refused: config hash differs from existing run_manifest.json")

    run_id = out_dir.name
    started_at = datetime.now(timezone.utc).isoformat()
    out_dir.mkdir(parents=True, exist_ok=True)
    for relative in [
        "maps",
        "episodes",
        "prompts",
        "skills/candidates",
        "skills/verified",
        "skills/reports",
        "capability",
        "figures",
    ]:
        (out_dir / relative).mkdir(parents=True, exist_ok=True)

    resolved = _resolved_config(config)
    (out_dir / "config_resolved.yaml").write_text(yaml.safe_dump(resolved, sort_keys=True), encoding="utf-8")
    partitions = make_seed_partitions(
        int(resolved["root_seed"]),
        {key: int(value) for key, value in resolved.get("seed_partition_sizes", {}).items()},
    )
    skills_by_group = _write_skill_sets(out_dir, run_id)
    prompt_rows = _write_prompt_manifest(out_dir)
    _write_map_files(out_dir, partitions)
    episode_rows = _write_episode_metrics(out_dir, partitions)
    capability_payload = _write_capability_outputs(out_dir, skills_by_group)
    _write_skill_trace(out_dir, run_id, skills_by_group)

    manifest = {
        "schema_version": 1,
        "run_id": run_id,
        "experiment_type": "skill_evolution",
        "mode": resolved["mode"],
        "mock_smoke": resolved["mode"] == "mock_smoke",
        "started_at": started_at,
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "config": str(config_path),
        "config_hash": config_hash,
        "code": _git_state(),
        "runtime_versions": _runtime_versions(),
        "seed_partitions": partitions.to_manifest(),
        "groups": list(GROUPS),
        "generator_version": resolved.get("generator_version", "fractal_percolation_v1"),
        "skill_runtime_version": resolved.get("skill_runtime_version", "restricted_skill_dsl_v1"),
        "verifier_version": resolved.get("verifier_version", "skill_verification_v1"),
        "prompt_template_hashes": {row["prompt_id"]: row["prompt_hash"] for row in prompt_rows},
        "candidate_verified_split": {
            group: {
                "candidate_skill_ids": [spec.skill_id for spec in specs if spec.status == "candidate"],
                "verified_skill_ids": [spec.skill_id for spec in specs if spec.status == "verified"],
            }
            for group, specs in skills_by_group.items()
        },
        "capability_summary": capability_payload["summary"],
        "output_file_checksums": _output_checksums(out_dir),
        "completion_status": "completed",
        "failure_reason": None,
    }
    (out_dir / "run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return manifest


def _resolved_config(config: dict[str, Any]) -> dict[str, Any]:
    experiment = dict(config.get("experiment", config))
    return {
        "mode": str(experiment.get("mode", "mock_smoke")),
        "root_seed": int(experiment.get("root_seed", 20260719)),
        "seed_partition_sizes": dict(
            experiment.get(
                "seed_partition_sizes",
                {
                    "map": 4,
                    "agent": 3,
                    "bootstrap": 5,
                    "test_map": 6,
                    "verify_map": 6,
                },
            )
        ),
        "generator_version": str(experiment.get("generator_version", "fractal_percolation_v1")),
        "skill_runtime_version": str(experiment.get("skill_runtime_version", "restricted_skill_dsl_v1")),
        "verifier_version": str(experiment.get("verifier_version", "skill_verification_v1")),
    }


def _write_skill_sets(out_dir: Path, run_id: str) -> dict[str, list[SkillSpec]]:
    rows: dict[str, list[SkillSpec]] = {group: [] for group in GROUPS}
    specs = {
        "handcrafted_skill": _fixture_skill("handcrafted_road_bias", "verified", "handcrafted"),
        "self_proposed_candidate": _fixture_skill("candidate_terrain_probe", "candidate", "fixture"),
        "self_proposed_verified": _fixture_skill("verified_terrain_probe", "verified", "fixture"),
        "prompt_skill": _fixture_skill("prompt_wrapped_route_skill", "verified", "fixture"),
    }
    for group, spec in specs.items():
        rows[group].append(spec)
        target = "verified" if spec.status == "verified" else "candidates"
        (out_dir / "skills" / target / f"{spec.skill_id}_{spec.version}.json").write_text(
            json.dumps({"schema_version": 1, "spec": spec.to_dict()}, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        if spec.status == "verified":
            report = {
                "schema_version": 1,
                "skill_id": spec.skill_id,
                "skill_version": spec.version,
                "decision": "verified",
                "report_type": "mock_smoke_fixture",
                "source_partition": "verify",
            }
            (out_dir / "skills" / "reports" / f"{spec.skill_id}_{spec.version}.json").write_text(
                json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True),
                encoding="utf-8",
            )
    return rows


def _fixture_skill(skill_id: str, status: str, proposer: str) -> SkillSpec:
    return SkillSpec.from_dict(
        {
            "schema_version": 1,
            "skill_id": skill_id,
            "version": "1.0.0",
            "status": status,
            "name": skill_id.replace("_", " ").title(),
            "description": "Deterministic fixture skill for M5 smoke experiments.",
            "problem_addressed": "Improve route choice on continuous terrain benchmark slices.",
            "source": {
                "proposer": proposer,
                "source_episode_ids": ["train/fixture/0"],
                "source_partition": "train",
            },
            "applicability": {"all": [{"feature": "current.terrain_band", "op": "in", "value": ["ROUGH", "VERY_ROUGH"]}]},
            "procedure": [
                {"op": "ESTIMATE", "target": "nearest_ore"},
                {"op": "PLAN_ROUTE", "avoid": "high_roughness"},
                {"op": "ACT", "action": "MOVE_RIGHT"},
            ],
            "budget": {
                "max_runtime_steps": 8,
                "max_environment_actions": 1,
                "max_nested_skill_depth": 0,
            },
            "objective": {"primary_metric": "road_net_payoff", "direction": "maximize"},
        }
    )


def _write_prompt_manifest(out_dir: Path) -> list[dict[str, Any]]:
    rows = [
        {
            "schema_version": 1,
            "prompt_id": SKILL_PROPOSER_PROMPT_ID,
            "prompt_version": SKILL_PROPOSER_PROMPT_VERSION,
            "prompt_hash": skill_proposer_prompt_hash(),
            "base_prompt_fixed": True,
            "used_by_groups": ["prompt_only", "self_proposed_candidate", "self_proposed_verified", "prompt_skill"],
        }
    ]
    _write_jsonl(out_dir / "prompts" / "prompt_manifest.jsonl", rows)
    return rows


def _write_map_files(out_dir: Path, partitions) -> None:
    manifest_rows = []
    diagnostics_rows = []
    for partition_name, partition in partitions.partitions.items():
        for index, seed in enumerate(partition.map_seeds):
            map_id = f"{partition_name}_map_{index}"
            manifest_rows.append({"schema_version": 1, "map_id": map_id, "partition": partition_name, "map_seed": seed})
            diagnostics_rows.append(
                {
                    "schema_version": 1,
                    "map_id": map_id,
                    "partition": partition_name,
                    "open_ratio": round(0.55 + (seed % 30) / 100.0, 3),
                    "connected": True,
                    "roughness_mean": round(0.2 + (seed % 50) / 100.0, 3),
                }
            )
    _write_jsonl(out_dir / "maps" / "map_manifest.jsonl", manifest_rows)
    _write_csv(out_dir / "maps" / "diagnostics.csv", diagnostics_rows)


def _write_episode_metrics(out_dir: Path, partitions) -> list[dict[str, Any]]:
    rows = []
    train_seeds = partitions.partition("train").map_seeds
    for group_index, group in enumerate(GROUPS):
        for episode_index, seed in enumerate(train_seeds):
            rows.append(
                {
                    "schema_version": 1,
                    "group": group,
                    "episode_id": f"{group}/train/{episode_index}",
                    "partition": "train",
                    "map_seed": seed,
                    "episode_reward": round(8.0 + group_index * 0.9 + (seed % 7) * 0.1, 3),
                    "ore_delivered": 1 + (group_index >= 2),
                    "road_usage_rate": round(0.1 + group_index * 0.08, 3),
                }
            )
    _write_csv(out_dir / "episodes" / "metrics.csv", rows)
    _write_jsonl(out_dir / "episodes" / "step_trace.jsonl", [])
    _write_jsonl(out_dir / "episodes" / "audit_trace.jsonl", [])
    return rows


def _write_capability_outputs(out_dir: Path, skills_by_group: dict[str, list[SkillSpec]]) -> dict[str, Any]:
    tasks = [
        CapabilityTask("open_small", "success", "maximize", 0.0, 1.0, 1.0),
        CapabilityTask("mixed_medium", "success", "maximize", 0.0, 1.0, 1.0),
        CapabilityTask("rough_large", "success", "maximize", 0.0, 1.0, 1.0),
    ]
    group_scores = {
        "no_skill": [0.38, 0.31, 0.24],
        "prompt_only": [0.44, 0.36, 0.28],
        "handcrafted_skill": [0.55, 0.47, 0.35],
        "self_proposed_candidate": [0.46, 0.38, 0.29],
        "self_proposed_verified": [0.62, 0.54, 0.42],
        "prompt_skill": [0.58, 0.50, 0.39],
    }
    benchmark_rows = []
    matrix_rows = []
    checkpoint_rows = []
    summary: dict[str, Any] = {"schema_version": 1, "mode": "mock_smoke", "by_group": {}}
    for index, group in enumerate(GROUPS):
        values = group_scores[group]
        benchmark_results = {
            task.task_id: {"success": values[task_index]}
            for task_index, task in enumerate(tasks)
        }
        verified_ids = [spec.skill_id for spec in skills_by_group[group] if spec.status == "verified"]
        effects = {
            skill_id: {
                task.task_id: round(0.03 + index * 0.02 + task_index * 0.01, 3)
                for task_index, task in enumerate(tasks)
            }
            for skill_id in verified_ids
        }
        retention = {task.task_id: max(0.0, values[task_index] - 0.05) for task_index, task in enumerate(tasks)}
        result = compute_capability(tasks, benchmark_results, verified_ids, effects, retention_results=retention)
        false_trigger_rate = 0.03 if verified_ids else 0.0
        summary["by_group"][group] = {
            **result.to_dict(),
            "false_trigger_rate": false_trigger_rate,
        }
        for task in tasks:
            benchmark_rows.append(
                {
                    "schema_version": 1,
                    "group": group,
                    "task_id": task.task_id,
                    "metric": task.metric,
                    "value": benchmark_results[task.task_id][task.metric],
                    "normalized": result.capability_vector[task.task_id],
                }
            )
        if result.skill_coverage_matrix:
            for row in result.skill_coverage_matrix:
                matrix_rows.append({"schema_version": 1, "group": group, **row})
        else:
            matrix_rows.append({"schema_version": 1, "group": group, "skill_id": "__none__", **{task.task_id: 0.0 for task in tasks}})
        checkpoint_rows.append(
            {
                "schema_version": 1,
                "checkpoint": index,
                "environment_stage_id": ["open_small", "open_small", "mixed_medium", "mixed_medium", "rough_large", "rough_large"][index],
                "group": group,
                "verified_skill_count": result.verified_skill_count,
                "capability_score": round(result.capability_score, 6),
                "ci_low": round(max(0.0, result.capability_score - 0.04), 6),
                "ci_high": round(min(1.0, result.capability_score + 0.04), 6),
                "false_trigger_rate": false_trigger_rate,
                "retention_score": round(float(result.retention_score or 0.0), 6),
                "promoted_skill_id": verified_ids[0] if verified_ids else "",
            }
        )
    _write_csv(out_dir / "capability" / "benchmark_results.csv", benchmark_rows)
    _write_csv(out_dir / "capability" / "capability_matrix.csv", matrix_rows)
    _write_csv(out_dir / "capability" / "checkpoints.csv", checkpoint_rows)
    (out_dir / "capability" / "capability_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return {"summary": summary, "checkpoints": checkpoint_rows}


def _write_skill_trace(out_dir: Path, run_id: str, skills_by_group: dict[str, list[SkillSpec]]) -> None:
    rows = []
    step = 0
    for group, specs in skills_by_group.items():
        for spec in specs:
            rows.append(
                {
                    "schema_version": 1,
                    "run_id": run_id,
                    "group": group,
                    "episode_id": f"{group}/mock/0",
                    "step": step,
                    "skill_id": spec.skill_id,
                    "skill_version": spec.version,
                    "spec_hash": spec.spec_hash,
                    "status": spec.status,
                    "event_type": "skill_available" if spec.status == "verified" else "candidate_recorded",
                }
            )
            step += 1
    _write_jsonl(out_dir / "skills" / "skill_trace.jsonl", rows)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _sha256(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _output_checksums(out_dir: Path) -> dict[str, str]:
    checksums = {}
    for path in sorted(out_dir.rglob("*")):
        if path.is_file() and path.name != "run_manifest.json":
            checksums[str(path.relative_to(out_dir)).replace("\\", "/")] = _file_sha256(path)
    return checksums


def _git_state() -> dict[str, Any]:
    def run(args: list[str]) -> str:
        try:
            return subprocess.check_output(args, text=True, stderr=subprocess.DEVNULL).strip()
        except Exception:  # noqa: BLE001
            return ""

    commit = run(["git", "rev-parse", "HEAD"])
    dirty = bool(run(["git", "status", "--porcelain"]))
    return {"commit": commit or None, "dirty": dirty}


def _runtime_versions() -> dict[str, Any]:
    versions = {"python": sys.version.split()[0], "platform": platform.platform()}
    try:
        import numpy

        versions["numpy"] = numpy.__version__
    except Exception:  # noqa: BLE001
        versions["numpy"] = None
    try:
        versions["pyyaml"] = yaml.__version__
    except Exception:  # noqa: BLE001
        versions["pyyaml"] = None
    return versions


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Skill-evolution M5 smoke experiment.")
    parser.add_argument("--config", default="configs/curriculum_self_evolution.yaml")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    manifest = run_skill_evolution_experiment(args.config, args.out)
    print(json.dumps({"run_manifest": str(Path(args.out) / "run_manifest.json"), "status": manifest["completion_status"]}))


if __name__ == "__main__":
    main()
