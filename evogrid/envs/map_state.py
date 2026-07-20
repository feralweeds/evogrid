"""Mutable map state for EvoGrid-Mine."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Set, Tuple

from evogrid.constants import Tile
from evogrid.envs.road_credit import RoadCreditTracker

Position = Tuple[int, int]


@dataclass
class MapState:
    grid: List[List[int]]
    base_pos: Position
    ore_positions: Set[Position]
    agent_pos: Position
    roughness: List[List[float]] | None = None
    map_id: str = ""
    static_diagnostics: Dict[str, Any] = field(default_factory=dict)
    has_ore: bool = False
    step_count: int = 0
    ore_delivered: int = 0
    total_reward: float = 0.0
    invalid_actions: int = 0
    num_mine: int = 0
    num_dig: int = 0
    num_build_road: int = 0
    changed_cells: Set[Position] = field(default_factory=set)
    built_roads: Set[Position] = field(default_factory=set)
    dug_cells: Set[Position] = field(default_factory=set)
    road_visited: Set[Position] = field(default_factory=set)
    road_credit_tracker: RoadCreditTracker = field(default_factory=RoadCreditTracker)
    visited_counts: Dict[Position, int] = field(default_factory=dict)
    action_history: List[int] = field(default_factory=list)
    mine_steps: List[int] = field(default_factory=list)
    shaping_action_steps: List[int] = field(default_factory=list)
    transport_steps: List[int] = field(default_factory=list)
    current_transport_steps: int = 0

    @property
    def height(self) -> int:
        return len(self.grid)

    @property
    def width(self) -> int:
        return len(self.grid[0]) if self.grid else 0

    def in_bounds(self, pos: Position) -> bool:
        row, col = pos
        return 0 <= row < self.height and 0 <= col < self.width

    def tile_at(self, pos: Position) -> Tile:
        row, col = pos
        return Tile(self.grid[row][col])

    def set_tile(self, pos: Position, tile: Tile) -> None:
        row, col = pos
        self.grid[row][col] = int(tile)

    def is_passable(self, pos: Position) -> bool:
        return self.in_bounds(pos) and self.tile_at(pos) != Tile.OBSTACLE

    def record_visit(self, pos: Position) -> None:
        self.visited_counts[pos] = self.visited_counts.get(pos, 0) + 1

    def to_dict(self) -> dict:
        return {
            "grid": [row[:] for row in self.grid],
            "roughness": None if self.roughness is None else [row[:] for row in self.roughness],
            "map_id": self.map_id,
            "static_diagnostics": self.static_diagnostics.copy(),
            "base_pos": list(self.base_pos),
            "ore_positions": [list(pos) for pos in sorted(self.ore_positions)],
            "agent_pos": list(self.agent_pos),
            "has_ore": self.has_ore,
            "step_count": self.step_count,
            "ore_delivered": self.ore_delivered,
            "total_reward": self.total_reward,
            "invalid_actions": self.invalid_actions,
            "num_mine": self.num_mine,
            "num_dig": self.num_dig,
            "num_build_road": self.num_build_road,
            "changed_cells": [list(pos) for pos in sorted(self.changed_cells)],
            "built_roads": [list(pos) for pos in sorted(self.built_roads)],
            "dug_cells": [list(pos) for pos in sorted(self.dug_cells)],
            "road_visited": [list(pos) for pos in sorted(self.road_visited)],
            "visited_counts": {f"{row},{col}": count for (row, col), count in sorted(self.visited_counts.items())},
            "action_history": self.action_history[:],
            "mine_steps": self.mine_steps[:],
            "shaping_action_steps": self.shaping_action_steps[:],
            "transport_steps": self.transport_steps[:],
            "current_transport_steps": self.current_transport_steps,
        }

    def to_observation(
        self,
        mode: str = "full_obs",
        local_view_radius: int = 4,
        observation_bins: tuple[float, ...] = (0.25, 0.5, 0.75),
        expose_continuous_roughness: bool = False,
    ) -> dict:
        if mode == "partial_obs":
            return self._partial_observation(
                local_view_radius,
                observation_bins=observation_bins,
                expose_continuous_roughness=expose_continuous_roughness,
            )
        return {
            "observation_mode": "full_obs",
            "grid": [row[:] for row in self.grid],
            "agent_pos": list(self.agent_pos),
            "base_pos": list(self.base_pos),
            "ore_positions": [list(pos) for pos in sorted(self.ore_positions)],
            "has_ore": self.has_ore,
            "step": self.step_count,
            "ore_delivered": self.ore_delivered,
        }

    def _partial_observation(
        self,
        radius: int,
        observation_bins: tuple[float, ...],
        expose_continuous_roughness: bool,
    ) -> dict:
        agent_row, agent_col = self.agent_pos
        origin = (agent_row - radius, agent_col - radius)
        local_view: list[list[int | None]] = []
        local_terrain_bands: list[list[str | None]] | None = [] if self.roughness is not None else None
        visible_tiles: list[dict] = []
        for row in range(agent_row - radius, agent_row + radius + 1):
            local_row: list[int | None] = []
            band_row: list[str | None] = []
            for col in range(agent_col - radius, agent_col + radius + 1):
                if self.in_bounds((row, col)):
                    tile = int(self.tile_at((row, col)))
                    local_row.append(tile)
                    item = {"pos": [row, col], "tile": tile}
                    if self.roughness is not None:
                        roughness = float(self.roughness[row][col])
                        band = terrain_band(roughness, observation_bins)
                        item["terrain_band"] = band
                        if expose_continuous_roughness:
                            item["roughness"] = roughness
                        band_row.append(band)
                    visible_tiles.append(item)
                else:
                    local_row.append(None)
                    if local_terrain_bands is not None:
                        band_row.append(None)
            local_view.append(local_row)
            if local_terrain_bands is not None:
                local_terrain_bands.append(band_row)
        observation = {
            "observation_mode": "partial_obs",
            "agent_pos": list(self.agent_pos),
            "base_pos": list(self.base_pos),
            "has_ore": self.has_ore,
            "step": self.step_count,
            "ore_delivered": self.ore_delivered,
            "local_view_radius": radius,
            "local_view_origin": list(origin),
            "local_view": local_view,
            "visible_tiles": visible_tiles,
            "recent_events": [],
        }
        if local_terrain_bands is not None:
            observation["local_terrain_bands"] = local_terrain_bands
        return observation


def terrain_band(roughness: float, bins: tuple[float, ...]) -> str:
    if roughness < bins[0]:
        return "SMOOTH"
    if len(bins) > 1 and roughness < bins[1]:
        return "NORMAL"
    if len(bins) > 2 and roughness < bins[2]:
        return "ROUGH"
    return "VERY_ROUGH"
