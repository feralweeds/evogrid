"""Partial-observation greedy baseline.

This agent is intentionally not the old full-info GreedyAgent. It only uses the
current local observation plus AgentMemory built from previous observations.
"""

from __future__ import annotations

import random

from evogrid.agents.base_agent import BaseAgent
from evogrid.agents.memory import AgentMemory
from evogrid.constants import MOVE_DELTAS, Action, Tile

Position = tuple[int, int]


class PartialGreedyAgent(BaseAgent):
    """A non-LLM baseline for partial-observation experiments."""

    def __init__(self, memory: AgentMemory | None = None):
        self.memory = memory or AgentMemory()
        self.rng = random.Random()
        self._explore_offset = 0

    def reset(self, seed: int | None = None) -> None:
        self.rng.seed(seed)
        self._explore_offset = int(seed or 0) % 4

    def act(self, obs: dict, info: dict) -> int:
        self.memory.update_from_observation(obs)
        agent_pos = _position(obs["agent_pos"])
        base_pos = _position(obs["base_pos"])

        if obs.get("has_ore") and agent_pos == base_pos:
            return int(Action.DROPOFF)
        if not obs.get("has_ore") and self._can_mine(obs):
            return int(Action.MINE)
        if obs.get("has_ore"):
            return self._move_toward(obs, base_pos) or self._least_visited_move(obs) or int(Action.NOOP)

        known_ore = self._nearest_known_ore(agent_pos)
        if known_ore is not None:
            move = self._move_toward(obs, known_ore)
            if move is not None:
                return move

        return self._least_visited_move(obs) or int(Action.NOOP)

    def observe_result(
        self,
        action: int,
        reward: float,
        obs: dict,
        info: dict,
        previous_info: dict | None = None,
    ) -> None:
        self.memory.update_from_result(action, reward, obs, info, previous_info)

    def _can_mine(self, obs: dict) -> bool:
        agent_pos = _position(obs["agent_pos"])
        if _visible_tile(obs, agent_pos) == int(Tile.ORE):
            return True
        for delta in MOVE_DELTAS.values():
            if _visible_tile(obs, _add(agent_pos, delta)) == int(Tile.ORE):
                return True
        return False

    def _nearest_known_ore(self, agent_pos: Position) -> Position | None:
        if not self.memory.seen_ore_locations:
            return None
        return min(self.memory.seen_ore_locations, key=lambda pos: _manhattan(agent_pos, pos))

    def _move_toward(self, obs: dict, target: Position) -> int | None:
        legal_moves = self._legal_moves(obs)
        if not legal_moves:
            return None
        agent_pos = _position(obs["agent_pos"])
        current_distance = _manhattan(agent_pos, target)
        best_action = None
        best_score = (current_distance, 10**9)
        for action_id in legal_moves:
            next_pos = _add(agent_pos, MOVE_DELTAS[Action(action_id)])
            distance = _manhattan(next_pos, target)
            visits = self.memory.visited_counts.get(next_pos, 0)
            score = (distance, visits)
            if score < best_score:
                best_action = action_id
                best_score = score
        return best_action

    def _least_visited_move(self, obs: dict) -> int | None:
        legal_moves = self._legal_moves(obs)
        if not legal_moves:
            return None
        agent_pos = _position(obs["agent_pos"])
        preferred = _rotated_moves(self._explore_offset + int(obs.get("step", 0)))
        return min(
            legal_moves,
            key=lambda action_id: (
                self.memory.visited_counts.get(_add(agent_pos, MOVE_DELTAS[Action(action_id)]), 0),
                preferred.index(Action(action_id)),
            ),
        )

    def _legal_moves(self, obs: dict) -> list[int]:
        agent_pos = _position(obs["agent_pos"])
        legal: list[int] = []
        for action in _rotated_moves(self._explore_offset):
            target = _add(agent_pos, MOVE_DELTAS[action])
            tile = _visible_tile(obs, target)
            if tile is not None and tile != int(Tile.OBSTACLE):
                legal.append(int(action))
        return legal


def _position(value) -> Position:
    return int(value[0]), int(value[1])


def _add(pos: Position, delta: tuple[int, int]) -> Position:
    return pos[0] + delta[0], pos[1] + delta[1]


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


def _rotated_moves(offset: int) -> list[Action]:
    moves = [Action.MOVE_RIGHT, Action.MOVE_DOWN, Action.MOVE_LEFT, Action.MOVE_UP]
    offset = offset % len(moves)
    return moves[offset:] + moves[:offset]


def _manhattan(left: Position, right: Position) -> int:
    return abs(left[0] - right[0]) + abs(left[1] - right[1])
