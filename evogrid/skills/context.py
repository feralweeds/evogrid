"""Observable context passed to Skill predicates/runtime."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


EVALUATOR_ONLY_KEYS = {
    "grid",
    "ore_positions",
    "route_rough_tile_count",
    "off_route_rough_tile_count",
    "positive_road_opportunity_count",
    "shortest_path_length",
    "minimum_cost_path_cost",
    "largest_component_fraction",
    "map_id",
    "static_diagnostics",
    "audit",
    "evaluator",
}


@dataclass(frozen=True)
class SkillContext:
    observation: dict[str, Any]
    observable_info: dict[str, Any]
    memory_summary: dict[str, Any] = field(default_factory=dict)
    route_plan: dict[str, Any] | None = None
    episode_budget: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_observable_inputs(
        cls,
        observation: dict[str, Any],
        info: dict[str, Any],
        memory_summary: dict[str, Any] | None = None,
        route_plan: dict[str, Any] | None = None,
        episode_budget: dict[str, Any] | None = None,
    ) -> "SkillContext":
        return cls(
            observation=_strip_evaluator_fields(observation),
            observable_info=_strip_evaluator_fields(info),
            memory_summary=_strip_evaluator_fields(memory_summary or {}),
            route_plan=None if route_plan is None else _strip_evaluator_fields(route_plan),
            episode_budget=_strip_evaluator_fields(episode_budget or {}),
        )

    def feature_root(self) -> dict[str, Any]:
        return {
            "current": _current_features(self.observation),
            "cargo": {"has_ore": bool(self.observation.get("has_ore"))},
            "route": self.route_plan or {},
            "memory": self.memory_summary,
            "local": _local_features(self.observation),
            "episode_budget": self.episode_budget,
            "info": self.observable_info,
        }


def _strip_evaluator_fields(value: Any) -> Any:
    if isinstance(value, dict):
        clean = {}
        for key, item in value.items():
            if str(key) in EVALUATOR_ONLY_KEYS:
                continue
            clean[str(key)] = _strip_evaluator_fields(item)
        return clean
    if isinstance(value, list):
        return [_strip_evaluator_fields(item) for item in value]
    return value


def _current_features(obs: dict[str, Any]) -> dict[str, Any]:
    agent_pos = tuple(obs.get("agent_pos", []))
    tile = None
    terrain_band = None
    for item in obs.get("visible_tiles", []):
        if tuple(item.get("pos", [])) == agent_pos:
            tile = item.get("tile")
            terrain_band = item.get("terrain_band")
            break
    return {
        "tile_type": tile,
        "terrain_band": terrain_band,
        "pos": list(agent_pos),
    }


def _local_features(obs: dict[str, Any]) -> dict[str, Any]:
    adjacent_obstacle_count = 0
    frontier_count = 0
    agent_pos = tuple(obs.get("agent_pos", []))
    for item in obs.get("visible_tiles", []):
        pos = tuple(item.get("pos", []))
        if len(pos) != 2 or len(agent_pos) != 2:
            continue
        distance = abs(pos[0] - agent_pos[0]) + abs(pos[1] - agent_pos[1])
        if distance == 1 and int(item.get("tile", -1)) == 3:
            adjacent_obstacle_count += 1
        if item.get("tile") is None:
            frontier_count += 1
    return {
        "adjacent_obstacle_count": adjacent_obstacle_count,
        "frontier_count": frontier_count,
    }
