"""Fixed curriculum schedule with replayable gate-promotion events."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from evogrid.curriculum.schemas import CurriculumConfig, CurriculumStage


@dataclass
class FixedScheduleCurriculum:
    config: CurriculumConfig
    stage_index: int = 0
    gate_history: list[dict[str, Any]] = field(default_factory=list)
    events: list[dict[str, Any]] = field(default_factory=list)

    def current_stage(self) -> CurriculumStage:
        return self.config.stages[self.stage_index]

    def record_gate_result(
        self,
        capability_score: float,
        *,
        benchmark_id: str | None = None,
        window_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        stage = self.current_stage()
        benchmark_id = benchmark_id or stage.gate_benchmark_id
        if benchmark_id != stage.gate_benchmark_id:
            raise ValueError(
                f"gate benchmark mismatch for {stage.stage_id}: expected {stage.gate_benchmark_id}, got {benchmark_id}"
            )
        event = {
            "schema_version": 1,
            "event_type": "gate_result",
            "stage_id": stage.stage_id,
            "benchmark_id": benchmark_id,
            "window_id": window_id or f"{stage.stage_id}_window_{len(self.gate_history)}",
            "capability_score": float(capability_score),
            "passed": float(capability_score) >= stage.promotion_rule.capability_score_gte,
            "metadata": dict(metadata or {}),
            "created_at": _now(),
        }
        self.gate_history.append(event)
        return event

    def should_promote(self) -> bool:
        if self.stage_index >= len(self.config.stages) - 1:
            return False
        stage = self.current_stage()
        needed = stage.promotion_rule.consecutive_windows
        if needed <= 0:
            return True
        recent = [
            event
            for event in self.gate_history
            if event["stage_id"] == stage.stage_id and event["benchmark_id"] == stage.gate_benchmark_id
        ][-needed:]
        return len(recent) == needed and all(bool(event["passed"]) for event in recent)

    def promote_if_ready(self, *, reason_code: str = "gate_passed") -> dict[str, Any] | None:
        if not self.should_promote():
            return None
        from_stage = self.current_stage()
        self.stage_index += 1
        to_stage = self.current_stage()
        event = {
            "schema_version": 1,
            "event_type": "stage_promotion",
            "from_stage": from_stage.stage_id,
            "to_stage": to_stage.stage_id,
            "to_parameters": dict(to_stage.env_family),
            "observable_evidence": {
                "gate_benchmark_id": from_stage.gate_benchmark_id,
                "consecutive_windows": from_stage.promotion_rule.consecutive_windows,
                "threshold": from_stage.promotion_rule.capability_score_gte,
            },
            "controller_version": self.config.controller_version,
            "reason_code": reason_code,
            "created_at": _now(),
        }
        self.events.append(event)
        return event

    def replayable_events(self) -> list[dict[str, Any]]:
        return [dict(event) for event in self.events]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()
