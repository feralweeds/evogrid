"""Trace objects emitted by Skill runtime."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class SkillTrace:
    schema_version: int
    run_id: str
    episode_id: str
    step: int
    skill_id: str
    skill_version: str
    spec_hash: str
    applicable: bool
    operations: list[dict[str, Any]] = field(default_factory=list)
    chosen_action: str | None = None
    termination: str = "not_started"
    observable_context_hash: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "episode_id": self.episode_id,
            "step": self.step,
            "skill_id": self.skill_id,
            "skill_version": self.skill_version,
            "spec_hash": self.spec_hash,
            "applicable": self.applicable,
            "operations": list(self.operations),
            "chosen_action": self.chosen_action,
            "termination": self.termination,
            "observable_context_hash": self.observable_context_hash,
        }
