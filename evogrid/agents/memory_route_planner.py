"""Route planning over the agent's observed memory map only."""

from __future__ import annotations

from dataclasses import dataclass
from heapq import heappop, heappush

from evogrid.agents.memory import AgentMemory
from evogrid.constants import MOVE_DELTAS, Action, Tile

Position = tuple[int, int]


@dataclass
class RoutePlan:
    action_id: int
    mode: str
    next_pos: list[int]
    target_pos: list[int]
    path: list[list[int]]
    cost: float
    reason: str


class MemoryMapRoutePlanner:
    """Find a low-level next action using only tiles in AgentMemory/current obs."""

    def __init__(
        self,
        road_cost: float = 0.5,
        ground_cost: float = 1.0,
        rough_cost: float = 2.0,
        unknown_cost: float = 3.0,
        dig_cost: float = 5.0,
        unknown_margin: int = 3,
    ):
        self.road_cost = road_cost
        self.ground_cost = ground_cost
        self.rough_cost = rough_cost
        self.unknown_cost = unknown_cost
        self.dig_cost = dig_cost
        self.unknown_margin = unknown_margin

    def plan_next_action(
        self,
        obs: dict,
        memory: AgentMemory,
        target: Position,
        allow_dig: bool = True,
        allow_unknown: bool = True,
    ) -> RoutePlan | None:
        start = _position(obs["agent_pos"])
        if start == target:
            return RoutePlan(
                action_id=int(Action.DROPOFF) if obs.get("has_ore") else int(Action.NOOP),
                mode="at_target",
                next_pos=list(start),
                target_pos=list(target),
                path=[list(start)],
                cost=0.0,
                reason="already at target",
            )

        path, cost = self._shortest_path(start, target, memory, allow_dig, allow_unknown)
        if not path or len(path) < 2:
            return None

        next_pos = path[1]
        next_tile = memory.seen_tiles.get(next_pos)
        if next_tile == int(Tile.OBSTACLE):
            if allow_dig and _adjacent(start, next_pos) and _visible_tile(obs, next_pos) == int(Tile.OBSTACLE):
                return RoutePlan(
                    action_id=int(Action.DIG),
                    mode="dig_known_obstacle",
                    next_pos=list(next_pos),
                    target_pos=list(target),
                    path=[list(pos) for pos in path],
                    cost=cost,
                    reason="planned route crosses an adjacent observed obstacle",
                )
            return None

        action = _action_from_step(start, next_pos)
        if action is None:
            return None
        visible_tile = _visible_tile(obs, next_pos)
        if visible_tile is not None and visible_tile == int(Tile.OBSTACLE):
            return None
        return RoutePlan(
            action_id=int(action),
            mode="follow_memory_route",
            next_pos=list(next_pos),
            target_pos=list(target),
            path=[list(pos) for pos in path],
            cost=cost,
            reason="follow lowest-cost route on observed memory map",
        )

    def _shortest_path(
        self,
        start: Position,
        target: Position,
        memory: AgentMemory,
        allow_dig: bool,
        allow_unknown: bool,
    ) -> tuple[list[Position], float]:
        bounds = _search_bounds(start, target, memory.seen_tiles.keys(), self.unknown_margin)
        frontier: list[tuple[float, int, Position]] = []
        heappush(frontier, (0.0, 0, start))
        came_from: dict[Position, Position | None] = {start: None}
        costs: dict[Position, float] = {start: 0.0}
        counter = 0

        while frontier:
            _, _, current = heappop(frontier)
            current_cost = costs.get(current, float("inf"))
            if current == target:
                return _reconstruct_path(came_from, target), current_cost

            for delta in MOVE_DELTAS.values():
                neighbor = (current[0] + delta[0], current[1] + delta[1])
                if not _in_bounds(neighbor, bounds):
                    continue
                step_cost = self._tile_cost(memory.seen_tiles.get(neighbor), allow_dig, allow_unknown)
                if step_cost is None:
                    continue
                new_cost = current_cost + step_cost
                if new_cost < costs.get(neighbor, float("inf")):
                    costs[neighbor] = new_cost
                    came_from[neighbor] = current
                    counter += 1
                    priority = new_cost + _manhattan(neighbor, target) * 0.01
                    heappush(frontier, (priority, counter, neighbor))
        return [], float("inf")

    def _tile_cost(self, tile: int | None, allow_dig: bool, allow_unknown: bool) -> float | None:
        if tile is None:
            return self.unknown_cost if allow_unknown else None
        if tile == int(Tile.OBSTACLE):
            return self.dig_cost if allow_dig else None
        if tile == int(Tile.ROAD):
            return self.road_cost
        if tile == int(Tile.ROUGH):
            return self.rough_cost
        return self.ground_cost


def _position(value) -> Position:
    return int(value[0]), int(value[1])


def _visible_tile(obs: dict, pos: Position) -> int | None:
    if obs.get("visible_tiles"):
        for item in obs["visible_tiles"]:
            if _position(item["pos"]) == pos:
                return int(item["tile"])
        return None
    return None


def _action_from_step(start: Position, next_pos: Position) -> Action | None:
    delta = (next_pos[0] - start[0], next_pos[1] - start[1])
    for action, action_delta in MOVE_DELTAS.items():
        if action_delta == delta:
            return action
    return None


def _search_bounds(
    start: Position,
    target: Position,
    known_positions,
    margin: int,
) -> tuple[int, int, int, int]:
    positions = list(known_positions) + [start, target]
    min_row = min(pos[0] for pos in positions) - margin
    max_row = max(pos[0] for pos in positions) + margin
    min_col = min(pos[1] for pos in positions) - margin
    max_col = max(pos[1] for pos in positions) + margin
    return min_row, max_row, min_col, max_col


def _in_bounds(pos: Position, bounds: tuple[int, int, int, int]) -> bool:
    min_row, max_row, min_col, max_col = bounds
    return min_row <= pos[0] <= max_row and min_col <= pos[1] <= max_col


def _reconstruct_path(came_from: dict[Position, Position | None], target: Position) -> list[Position]:
    path = [target]
    current = target
    while came_from[current] is not None:
        current = came_from[current]
        path.append(current)
    path.reverse()
    return path


def _adjacent(left: Position, right: Position) -> bool:
    return _manhattan(left, right) == 1


def _manhattan(left: Position, right: Position) -> int:
    return abs(left[0] - right[0]) + abs(left[1] - right[1])
