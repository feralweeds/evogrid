"""Adaptive curriculum controller constrained to train/gate evidence."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

from evogrid.envs.map_generation.seeding import derive_seed


ALLOWED_EVIDENCE_PARTITIONS = {"train", "gate"}
REASON_CODES = {"too_easy", "too_hard", "learning_frontier"}


@dataclass(frozen=True)
class AdaptiveParameterRule:
    name: str
    min_value: float
    max_value: float
    max_delta: float
    harder_delta_sign: int

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AdaptiveParameterRule":
        sign = int(data.get("harder_delta_sign", 1))
        if sign not in {-1, 1}:
            raise ValueError("harder_delta_sign must be -1 or 1")
        rule = cls(
            name=str(data["name"]),
            min_value=float(data["min_value"]),
            max_value=float(data["max_value"]),
            max_delta=float(data["max_delta"]),
            harder_delta_sign=sign,
        )
        if rule.min_value > rule.max_value:
            raise ValueError(f"{rule.name}: min_value must be <= max_value")
        if rule.max_delta < 0:
            raise ValueError(f"{rule.name}: max_delta must be non-negative")
        return rule

    def clamp(self, value: float) -> float:
        return min(self.max_value, max(self.min_value, value))

    def adjust(self, value: float, reason_code: str) -> float:
        if reason_code == "learning_frontier":
            return self.clamp(value)
        sign = self.harder_delta_sign if reason_code == "too_easy" else -self.harder_delta_sign
        return self.clamp(value + sign * self.max_delta)


@dataclass(frozen=True)
class AdaptiveEvidence:
    train_score: float
    gate_score: float
    improvement_rate: float = 0.0
    observed_partitions: tuple[str, ...] = ("train", "gate")
    window_id: str = ""
    sample_size: int = 0

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AdaptiveEvidence":
        partitions = tuple(str(item) for item in data.get("observed_partitions", ["train", "gate"]))
        forbidden = sorted(set(partitions) - ALLOWED_EVIDENCE_PARTITIONS)
        if forbidden:
            raise ValueError(f"adaptive curriculum cannot read partitions: {forbidden}")
        return cls(
            train_score=float(data.get("train_score", 0.0)),
            gate_score=float(data.get("gate_score", 0.0)),
            improvement_rate=float(data.get("improvement_rate", 0.0)),
            observed_partitions=partitions,
            window_id=str(data.get("window_id", "")),
            sample_size=int(data.get("sample_size", 0)),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "train_score": self.train_score,
            "gate_score": self.gate_score,
            "improvement_rate": self.improvement_rate,
            "observed_partitions": list(self.observed_partitions),
            "window_id": self.window_id,
            "sample_size": self.sample_size,
        }


@dataclass
class AdaptiveCurriculumController:
    root_seed: int
    controller_version: str = "adaptive_controller_v1"
    target_min: float = 0.45
    target_max: float = 0.75
    parameter_rules: dict[str, AdaptiveParameterRule] = field(default_factory=dict)
    events: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.target_min > self.target_max:
            raise ValueError("target_min must be <= target_max")
        if not self.parameter_rules:
            self.parameter_rules = default_parameter_rules()

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AdaptiveCurriculumController":
        rules = {
            str(item["name"]): AdaptiveParameterRule.from_dict(item)
            for item in data.get("parameter_rules", [])
        }
        return cls(
            root_seed=int(data.get("root_seed", 0)),
            controller_version=str(data.get("controller_version", "adaptive_controller_v1")),
            target_min=float(data.get("target_min", 0.45)),
            target_max=float(data.get("target_max", 0.75)),
            parameter_rules=rules,
        )

    def decide(
        self,
        stage_id: str,
        current_parameters: dict[str, Any],
        evidence: AdaptiveEvidence | dict[str, Any],
        *,
        decision_index: int | None = None,
    ) -> dict[str, Any]:
        evidence = evidence if isinstance(evidence, AdaptiveEvidence) else AdaptiveEvidence.from_dict(evidence)
        reason_code = self._reason_code(evidence)
        next_parameters = self._adjust_parameters(current_parameters, reason_code)
        if decision_index is None:
            decision_index = len(self.events)
        event = {
            "schema_version": 1,
            "event_type": "adaptive_curriculum_decision",
            "from_stage": str(stage_id),
            "to_parameters": next_parameters,
            "observable_evidence": evidence.to_dict(),
            "controller_version": self.controller_version,
            "reason_code": reason_code,
            "decision_seed": derive_seed(self.root_seed, "adaptive_curriculum", stage_id, decision_index),
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        self.events.append(event)
        return event

    def write_events_jsonl(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            "".join(json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n" for event in self.events),
            encoding="utf-8",
        )
        return path

    def _reason_code(self, evidence: AdaptiveEvidence) -> str:
        if evidence.gate_score > self.target_max:
            return "too_easy"
        if evidence.gate_score < self.target_min:
            return "too_hard"
        return "learning_frontier"

    def _adjust_parameters(self, parameters: dict[str, Any], reason_code: str) -> dict[str, Any]:
        adjusted = dict(parameters)
        for name, rule in self.parameter_rules.items():
            if name not in adjusted:
                continue
            adjusted[name] = _adjust_value(adjusted[name], rule, reason_code)
        return adjusted


def default_parameter_rules() -> dict[str, AdaptiveParameterRule]:
    return {
        "p_open": AdaptiveParameterRule(
            name="p_open",
            min_value=0.45,
            max_value=0.85,
            max_delta=0.03,
            harder_delta_sign=-1,
        ),
        "topology_hurst": AdaptiveParameterRule(
            name="topology_hurst",
            min_value=0.2,
            max_value=0.9,
            max_delta=0.05,
            harder_delta_sign=1,
        ),
        "terrain_hurst": AdaptiveParameterRule(
            name="terrain_hurst",
            min_value=0.1,
            max_value=0.9,
            max_delta=0.05,
            harder_delta_sign=1,
        ),
    }


def _adjust_value(value: Any, rule: AdaptiveParameterRule, reason_code: str) -> Any:
    if isinstance(value, list):
        return [_adjust_value(item, rule, reason_code) for item in value]
    if isinstance(value, tuple):
        return tuple(_adjust_value(item, rule, reason_code) for item in value)
    try:
        return round(rule.adjust(float(value), reason_code), 10)
    except (TypeError, ValueError):
        return value
