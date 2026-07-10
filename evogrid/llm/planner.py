"""Helpers that turn LLM decisions into primitive action ids."""

from __future__ import annotations

from evogrid.llm.schemas import LLMDecision


def decision_to_action_id(decision: LLMDecision) -> int | None:
    if decision.action_id is not None:
        return int(decision.action_id)
    preferred = decision.first_preferred_action_id()
    if preferred is not None:
        return preferred
    return None

