"""Schemas for fixed environment-family curricula."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class PromotionRule:
    capability_score_gte: float
    consecutive_windows: int

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PromotionRule":
        return cls(
            capability_score_gte=float(data.get("capability_score_gte", 0.0)),
            consecutive_windows=int(data.get("consecutive_windows", 1)),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "capability_score_gte": self.capability_score_gte,
            "consecutive_windows": self.consecutive_windows,
        }


@dataclass(frozen=True)
class CurriculumStage:
    stage_id: str
    env_family: dict[str, Any]
    train_budget_episodes: int
    gate_benchmark_id: str
    promotion_rule: PromotionRule

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CurriculumStage":
        required = ("stage_id", "env_family", "gate_benchmark_id")
        missing = [key for key in required if key not in data]
        if missing:
            raise ValueError(f"curriculum stage missing fields: {missing}")
        return cls(
            stage_id=str(data["stage_id"]),
            env_family=dict(data["env_family"]),
            train_budget_episodes=int(data.get("train_budget_episodes", 0)),
            gate_benchmark_id=str(data["gate_benchmark_id"]),
            promotion_rule=PromotionRule.from_dict(data.get("promotion_rule", {})),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage_id": self.stage_id,
            "env_family": dict(self.env_family),
            "train_budget_episodes": self.train_budget_episodes,
            "gate_benchmark_id": self.gate_benchmark_id,
            "promotion_rule": self.promotion_rule.to_dict(),
        }


@dataclass(frozen=True)
class CurriculumConfig:
    curriculum_id: str
    controller_version: str
    stages: list[CurriculumStage]
    allowed_partitions: tuple[str, ...] = ("train", "gate")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CurriculumConfig":
        stages = [CurriculumStage.from_dict(stage) for stage in data.get("stages", [])]
        if not stages:
            raise ValueError("curriculum requires at least one stage")
        allowed = tuple(str(item) for item in data.get("allowed_partitions", ["train", "gate"]))
        if "test" in allowed:
            raise ValueError("fixed curriculum must not read the test partition")
        return cls(
            curriculum_id=str(data.get("curriculum_id", "curriculum_self_evolution_v1")),
            controller_version=str(data.get("controller_version", "fixed_schedule_v1")),
            stages=stages,
            allowed_partitions=allowed,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "curriculum_id": self.curriculum_id,
            "controller_version": self.controller_version,
            "allowed_partitions": list(self.allowed_partitions),
            "stages": [stage.to_dict() for stage in self.stages],
        }
