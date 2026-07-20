"""Capability vectors and set-level Skill evaluation summaries."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class CapabilityTask:
    task_id: str
    metric: str
    direction: str
    floor: float
    reference: float
    weight: float


@dataclass(frozen=True)
class CapabilityResult:
    verified_skill_count: int
    capability_vector: dict[str, float]
    capability_score: float
    skill_coverage_matrix: list[dict[str, Any]]
    retention_score: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "verified_skill_count": self.verified_skill_count,
            "capability_vector": dict(self.capability_vector),
            "capability_score": self.capability_score,
            "skill_coverage_matrix": [dict(row) for row in self.skill_coverage_matrix],
            "retention_score": self.retention_score,
        }


def compute_capability(
    task_specs: list[CapabilityTask],
    benchmark_results: dict[str, dict[str, float]],
    verified_skill_ids: list[str],
    skill_task_effects: dict[str, dict[str, float]] | None = None,
    retention_results: dict[str, float] | None = None,
) -> CapabilityResult:
    if not task_specs:
        raise ValueError("task_specs: expected at least one task")
    total_weight = sum(task.weight for task in task_specs)
    if total_weight <= 0:
        raise ValueError("task_specs.weight: total weight must be positive")

    vector: dict[str, float] = {}
    weighted = 0.0
    for task in task_specs:
        if task.task_id not in benchmark_results:
            raise ValueError(f"missing benchmark result for task {task.task_id}")
        metrics = benchmark_results[task.task_id]
        if task.metric not in metrics:
            raise ValueError(f"missing metric {task.metric} for task {task.task_id}")
        normalized = _normalize(float(metrics[task.metric]), task)
        vector[task.task_id] = normalized
        weighted += task.weight * normalized

    coverage = []
    effects = skill_task_effects or {}
    for skill_id in verified_skill_ids:
        row = {"skill_id": skill_id}
        for task in task_specs:
            if skill_id not in effects or task.task_id not in effects[skill_id]:
                raise ValueError(f"missing skill coverage for {skill_id} on {task.task_id}")
            row[task.task_id] = float(effects[skill_id][task.task_id])
        coverage.append(row)

    retention_score = None
    if retention_results:
        missing = [task.task_id for task in task_specs if task.task_id not in retention_results]
        if missing:
            raise ValueError(f"missing retention tasks: {missing}")
        retention_score = sum(float(retention_results[task.task_id]) for task in task_specs) / len(task_specs)

    return CapabilityResult(
        verified_skill_count=len(verified_skill_ids),
        capability_vector=vector,
        capability_score=weighted / total_weight,
        skill_coverage_matrix=coverage,
        retention_score=retention_score,
    )


def _normalize(value: float, task: CapabilityTask) -> float:
    if task.reference == task.floor:
        raise ValueError(f"task {task.task_id}: reference and floor must differ")
    if task.direction == "maximize":
        return (value - task.floor) / (task.reference - task.floor)
    if task.direction == "minimize":
        return (task.floor - value) / (task.floor - task.reference)
    raise ValueError(f"task {task.task_id}: unsupported direction")
