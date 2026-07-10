"""Greedy baseline agent with shortest-path movement."""

from __future__ import annotations

from collections import deque
from typing import Iterable, Tuple

from evogrid.agents.base_agent import BaseAgent
from evogrid.constants import Action, Tile

Position = Tuple[int, int]


class GreedyAgent(BaseAgent):
    """Moves to ore, mines, then returns to base."""

    def act(self, obs: dict, info: dict) -> int:
        agent_pos = tuple(obs["agent_pos"])
        base_pos = tuple(obs["base_pos"])
        ore_positions = [tuple(pos) for pos in obs["ore_positions"]]
        has_ore = bool(obs["has_ore"])

        if has_ore:
            if agent_pos == base_pos:
                return int(Action.DROPOFF)
            next_pos = _next_step(obs["grid"], agent_pos, [base_pos])
            return _move_action(agent_pos, next_pos)

        if agent_pos in ore_positions or any(pos in ore_positions for pos in _neighbors(agent_pos)):
            return int(Action.MINE)

        next_pos = _next_step(obs["grid"], agent_pos, ore_positions)
        return _move_action(agent_pos, next_pos)


def _neighbors(pos: Position) -> Iterable[Position]:
    row, col = pos
    yield (row - 1, col)
    yield (row + 1, col)
    yield (row, col - 1)
    yield (row, col + 1)


def _passable(grid: list[list[int]], pos: Position) -> bool:
    row, col = pos
    return (
        0 <= row < len(grid)
        and 0 <= col < len(grid[0])
        and Tile(grid[row][col]) != Tile.OBSTACLE
    )


def _next_step(grid: list[list[int]], start: Position, goals: list[Position]) -> Position | None:
    goal_set = set(goals)
    queue = deque([start])
    came_from: dict[Position, Position | None] = {start: None}

    while queue:
        current = queue.popleft()
        if current in goal_set:
            return _first_step(came_from, start, current)
        for nxt in _neighbors(current):
            if nxt in came_from or not _passable(grid, nxt):
                continue
            came_from[nxt] = current
            queue.append(nxt)
    return None


def _first_step(
    came_from: dict[Position, Position | None],
    start: Position,
    goal: Position,
) -> Position | None:
    if goal == start:
        return start
    current = goal
    previous = came_from[current]
    while previous is not None and previous != start:
        current = previous
        previous = came_from[current]
    return current


def _move_action(current: Position, target: Position | None) -> int:
    if target is None or target == current:
        return int(Action.NOOP)
    row, col = current
    target_row, target_col = target
    if target_row == row - 1 and target_col == col:
        return int(Action.MOVE_UP)
    if target_row == row + 1 and target_col == col:
        return int(Action.MOVE_DOWN)
    if target_row == row and target_col == col - 1:
        return int(Action.MOVE_LEFT)
    if target_row == row and target_col == col + 1:
        return int(Action.MOVE_RIGHT)
    return int(Action.NOOP)

