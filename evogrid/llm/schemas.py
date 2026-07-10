"""Structured LLM decision schema."""

from __future__ import annotations

from dataclasses import dataclass, field

from evogrid.constants import Action, action_from_name


@dataclass
class LLMDecision:
    mode: str = "action"
    action_id: int | None = None
    action: str | None = None
    subgoal: str | None = None
    target_cells: list[list[int]] = field(default_factory=list)
    preferred_actions: list[str] = field(default_factory=list)
    stop_condition: str | None = None
    reason: str = ""
    confidence: float = 0.0

    @classmethod
    def from_dict(cls, data: dict) -> "LLMDecision":
        preferred_actions = data.get("preferred_actions") or []
        if isinstance(preferred_actions, str):
            preferred_actions = [preferred_actions]
        target_cells = data.get("target_cells") or []
        confidence = data.get("confidence", 0.0) or 0.0
        try:
            confidence = float(confidence)
        except (TypeError, ValueError):
            confidence = 0.0

        decision = cls(
            mode=str(data.get("mode", "action")),
            action_id=data.get("action_id"),
            action=data.get("action"),
            subgoal=data.get("subgoal"),
            target_cells=target_cells,
            preferred_actions=preferred_actions,
            stop_condition=data.get("stop_condition"),
            reason=str(data.get("reason", "")),
            confidence=confidence,
        )
        if decision.action_id is None and decision.action:
            decision.action_id = int(action_from_name(decision.action))
        if decision.action_id is not None:
            decision.action_id = int(Action(int(decision.action_id)))
        return decision

    def first_preferred_action_id(self) -> int | None:
        for name in self.preferred_actions:
            try:
                return int(action_from_name(name))
            except ValueError:
                continue
        return None
