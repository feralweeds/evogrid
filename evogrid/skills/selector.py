"""Skill selection over observable context."""

from __future__ import annotations

from dataclasses import dataclass

from evogrid.skills.context import SkillContext
from evogrid.skills.predicates import evaluate_predicate
from evogrid.skills.schemas import SkillRecord


@dataclass(frozen=True)
class SkillSelection:
    record: SkillRecord | None
    reason: str


class SkillSelector:
    def __init__(self, allow_candidates: bool = False):
        self.allow_candidates = allow_candidates

    def select(self, records: list[SkillRecord], context: SkillContext) -> SkillSelection:
        applicable: list[SkillRecord] = []
        for record in records:
            if record.spec.status == "candidate" and not self.allow_candidates:
                continue
            if record.spec.status not in {"verified", "candidate"}:
                continue
            result = evaluate_predicate(record.spec.applicability, context)
            if result.ok and result.applicable:
                applicable.append(record)
        if not applicable:
            return SkillSelection(None, "no_applicable_skill")
        applicable.sort(key=lambda record: (-int(record.spec.budget.get("priority", 0) or 0), record.spec.skill_id, record.spec.version))
        return SkillSelection(applicable[0], "selected")
