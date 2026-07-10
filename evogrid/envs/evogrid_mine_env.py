"""A minimal mutable 2D mining environment."""

from __future__ import annotations

from collections import deque
from copy import deepcopy
from typing import Iterable, Tuple

from evogrid.constants import Action, MOVE_DELTAS, Tile
from evogrid.envs.map_builder import build_fixed_map
from evogrid.envs.map_state import MapState, Position
from evogrid.envs.metrics import collect_metrics
from evogrid.envs.reward import RewardConfig


class EvoGridMineEnv:
    """Gymnasium-style environment without requiring Gymnasium as a dependency."""

    metadata = {"render_modes": ["ansi"]}

    def __init__(self, config: dict | None = None):
        self.config = config or {}
        self.env_config = self.config.get("env", self.config)
        self.max_steps = int(self.env_config.get("max_steps", 500))
        observation_config = self.env_config.get("observation", {})
        self.observation_mode = str(
            observation_config.get("mode") or self.env_config.get("observation_mode") or "full_obs"
        )
        self.local_view_radius = int(
            observation_config.get("local_view_radius") or self.env_config.get("local_view_radius") or 4
        )
        shaping = self.env_config.get("shaping", {})
        self.allow_dig = bool(shaping.get("allow_dig", True))
        self.allow_build_road = bool(shaping.get("allow_build_road", True))
        self.reset_after_dropoff = bool(shaping.get("reset_after_dropoff", False))
        self.rewards = RewardConfig.from_config(self.config)
        self.initial_grid: list[list[int]] = []
        self.state: MapState | None = None

    def reset(self, seed: int | None = None) -> tuple[dict, dict]:
        grid, base_pos, ore_positions = build_fixed_map(self.config, seed=seed)
        self.initial_grid = deepcopy(grid)
        self.state = MapState(
            grid=grid,
            base_pos=base_pos,
            ore_positions=ore_positions,
            agent_pos=base_pos,
        )
        self.state.record_visit(base_pos)
        return self._obs(), self._info()

    def step(self, action: int | Action) -> tuple[dict, float, bool, bool, dict]:
        if self.state is None:
            raise RuntimeError("Call reset() before step().")

        action = Action(int(action))
        reward = 0.0
        valid = True
        was_carrying = self.state.has_ore

        if action in MOVE_DELTAS:
            reward, valid = self._move(action)
        elif action == Action.MINE:
            reward, valid = self._mine()
        elif action == Action.DIG:
            reward, valid = self._dig()
        elif action == Action.BUILD_ROAD:
            reward, valid = self._build_road()
        elif action == Action.DROPOFF:
            reward, valid = self._dropoff()
        elif action == Action.NOOP:
            reward = self.rewards.noop
        else:
            reward, valid = self.rewards.invalid_action, False

        if not valid:
            self.state.invalid_actions += 1

        self.state.action_history.append(int(action))
        self.state.step_count += 1
        if was_carrying and self.state.has_ore:
            self.state.current_transport_steps += 1
        self.state.total_reward += reward
        self.state.record_visit(self.state.agent_pos)

        terminated = False
        truncated = self.state.step_count >= self.max_steps
        return self._obs(), reward, terminated, truncated, self._info()

    def render(self) -> str:
        if self.state is None:
            return ""
        rows = []
        for row_idx, row in enumerate(self.state.grid):
            chars = []
            for col_idx, value in enumerate(row):
                pos = (row_idx, col_idx)
                if pos == self.state.agent_pos:
                    chars.append("A")
                else:
                    chars.append(_tile_char(Tile(value)))
            rows.append("".join(chars))
        return "\n".join(rows)

    def _move(self, action: Action) -> tuple[float, bool]:
        assert self.state is not None
        delta_row, delta_col = MOVE_DELTAS[action]
        row, col = self.state.agent_pos
        target = (row + delta_row, col + delta_col)
        if not self.state.is_passable(target):
            return self.rewards.invalid_action, False

        self.state.agent_pos = target
        tile = self.state.tile_at(target)
        if tile == Tile.ROAD and target in self.state.built_roads:
            self.state.road_visited.add(target)
            self.state.road_credit_tracker.record_use(target, self.state.step_count)
        if tile == Tile.ROAD:
            return self.rewards.move_road, True
        if tile == Tile.ROUGH:
            return self.rewards.move_rough, True
        return self.rewards.move_ground, True

    def _mine(self) -> tuple[float, bool]:
        assert self.state is not None
        if self.state.has_ore:
            return self.rewards.invalid_action, False
        if self.state.agent_pos in self.state.ore_positions:
            self.state.has_ore = True
            self.state.current_transport_steps = 0
            self.state.num_mine += 1
            self.state.mine_steps.append(self.state.step_count)
            return self.rewards.mine, True
        for pos in self._adjacent_positions(self.state.agent_pos):
            if pos in self.state.ore_positions:
                self.state.has_ore = True
                self.state.current_transport_steps = 0
                self.state.num_mine += 1
                self.state.mine_steps.append(self.state.step_count)
                return self.rewards.mine, True
        return self.rewards.invalid_action, False

    def _dig(self) -> tuple[float, bool]:
        assert self.state is not None
        if not self.allow_dig:
            return self.rewards.invalid_action, False
        target = self._find_adjacent_tile(Tile.OBSTACLE)
        if target is None:
            return self.rewards.invalid_action, False
        self.state.set_tile(target, Tile.GROUND)
        self.state.changed_cells.add(target)
        self.state.dug_cells.add(target)
        self.state.num_dig += 1
        self.state.shaping_action_steps.append(self.state.step_count)
        return self.rewards.dig, True

    def _build_road(self) -> tuple[float, bool]:
        assert self.state is not None
        if not self.allow_build_road:
            return self.rewards.invalid_action, False
        pos = self.state.agent_pos
        tile = self.state.tile_at(pos)
        if tile in {Tile.BASE, Tile.ORE, Tile.OBSTACLE, Tile.ROAD}:
            return self.rewards.invalid_action, False
        self.state.road_credit_tracker.record_build(
            position=pos,
            original_tile=tile,
            original_move_cost=self._move_cost(tile),
            road_move_cost=self._move_cost(Tile.ROAD),
            build_cost=max(0.0, -float(self.rewards.build_road)),
            build_step=self.state.step_count,
        )
        self.state.set_tile(pos, Tile.ROAD)
        self.state.changed_cells.add(pos)
        self.state.built_roads.add(pos)
        self.state.num_build_road += 1
        self.state.shaping_action_steps.append(self.state.step_count)
        return self.rewards.build_road, True

    def _dropoff(self) -> tuple[float, bool]:
        assert self.state is not None
        if self.state.agent_pos != self.state.base_pos or not self.state.has_ore:
            return self.rewards.invalid_action, False
        self.state.has_ore = False
        self.state.ore_delivered += 1
        self.state.transport_steps.append(self.state.current_transport_steps)
        self.state.current_transport_steps = 0
        if self.reset_after_dropoff:
            self._clear_shaping()
        return self.rewards.dropoff + self.rewards.dropoff_action, True

    def _clear_shaping(self) -> None:
        assert self.state is not None
        for row, col in self.state.built_roads | self.state.dug_cells:
            self.state.grid[row][col] = self.initial_grid[row][col]
        self.state.changed_cells.clear()
        self.state.built_roads.clear()
        self.state.dug_cells.clear()
        self.state.road_visited.clear()
        self.state.road_credit_tracker.clear()

    def _adjacent_positions(self, pos: Position) -> Iterable[Position]:
        row, col = pos
        yield (row - 1, col)
        yield (row + 1, col)
        yield (row, col - 1)
        yield (row, col + 1)

    def _find_adjacent_tile(self, tile: Tile) -> Position | None:
        assert self.state is not None
        for pos in self._adjacent_positions(self.state.agent_pos):
            if self.state.in_bounds(pos) and self.state.tile_at(pos) == tile:
                return pos
        return None

    def _move_cost(self, tile: Tile) -> float:
        if tile == Tile.ROAD:
            return max(0.0, -float(self.rewards.move_road))
        if tile == Tile.ROUGH:
            return max(0.0, -float(self.rewards.move_rough))
        return max(0.0, -float(self.rewards.move_ground))

    def _obs(self) -> dict:
        assert self.state is not None
        return self.state.to_observation(
            mode=self.observation_mode,
            local_view_radius=self.local_view_radius,
        )

    def _info(self) -> dict:
        assert self.state is not None
        metrics = collect_metrics(self.state).to_dict()
        metrics.update(self._map_diagnostics())
        map_summary = {
            "agent_pos": list(self.state.agent_pos),
            "base_pos": list(self.state.base_pos),
            "has_ore": self.state.has_ore,
            "step": self.state.step_count,
        }
        if self.observation_mode != "partial_obs":
            map_summary["ore_positions"] = [list(pos) for pos in sorted(self.state.ore_positions)]
        metrics["map_summary"] = map_summary
        return metrics

    def _map_diagnostics(self) -> dict:
        assert self.state is not None
        grid = self.initial_grid or self.state.grid
        rough_cells = {
            (row, col)
            for row, values in enumerate(grid)
            for col, value in enumerate(values)
            if Tile(value) == Tile.ROUGH
        }
        buildable_cells = {
            (row, col)
            for row, values in enumerate(grid)
            for col, value in enumerate(values)
            if Tile(value) in {Tile.GROUND, Tile.ROUGH}
        }
        path = _shortest_path_to_any_ore(grid, self.state.base_pos, self.state.ore_positions)
        route_cells = set(path)
        route_rough_count = len(route_cells & rough_cells)
        off_route_rough_count = len(rough_cells - route_cells)
        return {
            "rough_tile_count": len(rough_cells),
            "buildable_tile_count": len(buildable_cells),
            "shaping_opportunity_count": len(buildable_cells),
            "transport_corridor_length": max(0, len(path) - 1),
            "route_rough_tile_count": route_rough_count,
            "off_route_rough_tile_count": off_route_rough_count,
            "positive_road_opportunity_count": route_rough_count,
        }


def _tile_char(tile: Tile) -> str:
    from evogrid.constants import TILE_CHARS

    return TILE_CHARS[tile]


def _shortest_path_to_any_ore(
    grid: list[list[int]],
    start: Position,
    ore_positions: set[Position],
) -> list[Position]:
    if not ore_positions:
        return []
    queue = deque([start])
    parents: dict[Position, Position | None] = {start: None}
    while queue:
        current = queue.popleft()
        if current in ore_positions:
            return _reconstruct_path(parents, current)
        row, col = current
        for nxt in ((row - 1, col), (row + 1, col), (row, col - 1), (row, col + 1)):
            if nxt in parents:
                continue
            next_row, next_col = nxt
            if not (0 <= next_row < len(grid) and 0 <= next_col < len(grid[0])):
                continue
            if Tile(grid[next_row][next_col]) == Tile.OBSTACLE:
                continue
            parents[nxt] = current
            queue.append(nxt)
    return []


def _reconstruct_path(parents: dict[Position, Position | None], goal: Position) -> list[Position]:
    path = []
    current: Position | None = goal
    while current is not None:
        path.append(current)
        current = parents[current]
    path.reverse()
    return path
