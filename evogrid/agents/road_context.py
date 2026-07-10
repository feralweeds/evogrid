"""Utilities for attaching agent-visible context to road payoff records."""

from __future__ import annotations

from evogrid.constants import Action

Position = tuple[int, int]


def contextualize_road_credit_records(records: list[dict], trace: list[dict]) -> list[dict]:
    context_by_key = {}
    for item in trace:
        if _trace_action(item) != "BUILD_ROAD":
            continue
        opportunity = item.get("shaping_opportunity") or item.get("prompt_shaping_opportunity") or {}
        pos = _optional_position(item.get("agent_pos") or opportunity.get("position"))
        if pos is None:
            continue
        step = int(item.get("step", 0) or 0)
        route_context = opportunity.get("route_context", {}) or {}
        memory_evidence = opportunity.get("memory_evidence", {}) or {}
        context_by_key[(pos, step)] = {
            "route_on_build": bool(route_context.get("on_current_route")),
            "build_mode": route_context.get("mode"),
            "route_remaining_length": route_context.get("route_remaining_length"),
            "known_as_transport_corridor": bool(memory_evidence.get("known_as_transport_corridor")),
            "observed_visit_count_on_build": int(memory_evidence.get("observed_visit_count", 0) or 0),
            "build_decision_source": item.get("build_decision_source") or "unknown",
        }

    contextualized = []
    for record in records:
        clean = dict(record)
        pos = _optional_position(clean.get("position"))
        build_step = clean.get("build_step")
        if pos is not None and build_step is not None:
            clean.update(context_by_key.get((pos, int(build_step)), {}))
        contextualized.append(clean)
    return contextualized


def _trace_action(item: dict) -> str:
    action = item.get("action") or item.get("chosen_action_name")
    if action:
        return str(action)
    action_id = item.get("action_id", item.get("chosen_action"))
    if action_id is None:
        return ""
    try:
        return Action(int(action_id)).name
    except ValueError:
        return str(action_id)


def _optional_position(value) -> Position | None:
    if value is None:
        return None
    return int(value[0]), int(value[1])
