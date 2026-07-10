"""Candidate environment-shaping evidence for agent decisions."""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil

from evogrid.agents.memory import AgentMemory
from evogrid.agents.memory_route_planner import RoutePlan
from evogrid.agents.road_learning import RoadLearningModule
from evogrid.constants import Action, Tile

Position = tuple[int, int]


@dataclass(frozen=True)
class RoadEconomics:
    build_cost: float = 0.1
    ground_move_cost: float = 0.01
    rough_move_cost: float = 0.05
    road_move_cost: float = 0.0

    def move_cost(self, tile: int) -> float:
        if tile == int(Tile.ROUGH):
            return self.rough_move_cost
        if tile == int(Tile.ROAD):
            return self.road_move_cost
        return self.ground_move_cost


class ShapingOpportunityBuilder:
    """Build non-binding road-building evidence from observed state only."""

    def __init__(self, economics: RoadEconomics | None = None):
        self.economics = economics or RoadEconomics()

    def build(
        self,
        obs: dict,
        info: dict,
        memory: AgentMemory,
        route_plan: RoutePlan | None = None,
        mode: str = "EXPLORE",
    ) -> dict:
        position = _position(obs["agent_pos"])
        current_tile = _visible_tile(obs, position)
        if current_tile not in {int(Tile.GROUND), int(Tile.ROUGH)}:
            return {
                "available": False,
                "reason": "current tile is not road-buildable",
                "position": list(position),
                "current_tile": _tile_name(current_tile),
                "constraints": _constraints(),
            }

        original_move_cost = self.economics.move_cost(current_tile)
        saving_per_use = original_move_cost - self.economics.road_move_cost
        road_learning = RoadLearningModule.from_records(memory.road_credit_records)
        memory_evidence = {
            "observed_visit_count": int(memory.visited_counts.get(position, 0)),
            "built_before": position in memory.built_roads,
            "known_as_transport_corridor": _on_route(position, route_plan) or bool(obs.get("has_ore")),
        }
        route_context = _route_context(route_plan, mode)
        return {
            "available": True,
            "candidate_action": Action.BUILD_ROAD.name,
            "position": list(position),
            "current_tile": _tile_name(current_tile),
            "cost": {
                "build_cost": self.economics.build_cost,
                "original_move_cost": original_move_cost,
                "road_move_cost": self.economics.road_move_cost,
                "saving_per_use": saving_per_use,
                "break_even_uses": _break_even_uses(self.economics.build_cost, saving_per_use),
            },
            "memory_evidence": memory_evidence,
            "route_context": route_context,
            "history_stats": _history_stats(memory.road_credit_records, current_tile),
            "learned_estimate": road_learning.estimate(
                current_tile,
                context={"memory_evidence": memory_evidence, "route_context": route_context},
            ),
            "constraints": _constraints(),
        }


def _route_context(route_plan: RoutePlan | None, mode: str) -> dict:
    if route_plan is None:
        return {
            "has_route_plan": False,
            "mode": mode,
            "on_current_route": False,
            "route_target": None,
            "route_remaining_length": None,
        }
    return {
        "has_route_plan": True,
        "mode": mode,
        "on_current_route": True,
        "route_target": route_plan.target_pos,
        "route_remaining_length": len(route_plan.path),
    }


def _history_stats(records: list[dict], current_tile: int) -> dict:
    similar = [record for record in records if int(record.get("original_tile", -1)) == current_tile]
    if not similar:
        return {
            "similar_tile_build_count": 0,
            "similar_tile_mean_payoff": 0.0,
            "similar_tile_positive_rate": 0.0,
            "history_scope": "observed_road_credit_records_only",
        }
    payoffs = [float(record.get("net_payoff", 0.0) or 0.0) for record in similar]
    positives = sum(1 for payoff in payoffs if payoff > 0.0)
    return {
        "similar_tile_build_count": len(similar),
        "similar_tile_mean_payoff": sum(payoffs) / len(payoffs),
        "similar_tile_positive_rate": positives / len(similar),
        "history_scope": "observed_road_credit_records_only",
    }


def _constraints() -> dict:
    return {
        "uses_hidden_map": False,
        "uses_future_truth": False,
        "auto_execute": False,
    }


def _break_even_uses(build_cost: float, saving_per_use: float) -> int | None:
    if saving_per_use <= 0.0:
        return None
    return int(ceil(build_cost / saving_per_use))


def _on_route(position: Position, route_plan: RoutePlan | None) -> bool:
    if route_plan is None:
        return False
    return list(position) in route_plan.path


def _position(value) -> Position:
    return int(value[0]), int(value[1])


def _visible_tile(obs: dict, pos: Position) -> int | None:
    if obs.get("visible_tiles"):
        for item in obs["visible_tiles"]:
            if _position(item["pos"]) == pos:
                return int(item["tile"])
        return None
    if "local_view" in obs:
        origin_row, origin_col = obs.get("local_view_origin", [0, 0])
        row_idx = pos[0] - int(origin_row)
        col_idx = pos[1] - int(origin_col)
        local_view = obs.get("local_view", [])
        if 0 <= row_idx < len(local_view) and 0 <= col_idx < len(local_view[row_idx]):
            value = local_view[row_idx][col_idx]
            return None if value is None else int(value)
        return None
    if "grid" in obs:
        grid = obs["grid"]
        row, col = pos
        if 0 <= row < len(grid) and 0 <= col < len(grid[0]):
            return int(grid[row][col])
    return None


def _tile_name(tile: int | None) -> str:
    if tile is None:
        return "UNKNOWN"
    return Tile(int(tile)).name
