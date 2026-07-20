"""Agent-side memory built only from observed tiles and action outcomes."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

from evogrid.agents.road_learning import RoadLearningModule
from evogrid.constants import Action, Tile

Position = tuple[int, int]


@dataclass
class AgentMemory:
    seen_tiles: dict[Position, int] = field(default_factory=dict)
    seen_terrain_bands: dict[Position, str] = field(default_factory=dict)
    seen_ore_locations: set[Position] = field(default_factory=set)
    seen_obstacles: set[Position] = field(default_factory=set)
    seen_rough_tiles: set[Position] = field(default_factory=set)
    seen_roads: set[Position] = field(default_factory=set)
    visited_counts: dict[Position, int] = field(default_factory=dict)
    built_roads: set[Position] = field(default_factory=set)
    dug_cells: set[Position] = field(default_factory=set)
    road_credit_records: list[dict] = field(default_factory=list)
    failed_actions: list[dict] = field(default_factory=list)
    reflections: list[dict] = field(default_factory=list)
    update_count: int = 0
    first_ore_seen_step: int | None = None
    first_mine_step: int | None = None
    _road_credit_index: dict[tuple[int, Position, int], int] = field(default_factory=dict)
    _episode_index: int = 0
    _last_observed_step: int | None = None

    def update_from_observation(self, obs: dict) -> None:
        agent_pos = _optional_position(obs.get("agent_pos"))
        step = _optional_int(obs.get("step"))
        if step is not None:
            if self._last_observed_step is not None and step < self._last_observed_step:
                self._episode_index += 1
            self._last_observed_step = step
        if agent_pos is not None:
            self.visited_counts[agent_pos] = self.visited_counts.get(agent_pos, 0) + 1

        for pos, tile, terrain_band in _iter_observed_tiles(obs):
            previous_tile = self.seen_tiles.get(pos)
            if previous_tile != tile:
                self.update_count += 1
            self.seen_tiles[pos] = tile
            if terrain_band is not None:
                self.seen_terrain_bands[pos] = terrain_band
            self._sync_tile_sets(pos, tile)
            if tile == int(Tile.ORE) and self.first_ore_seen_step is None:
                self.first_ore_seen_step = step

    def update_from_result(
        self,
        action: int | Action,
        reward: float,
        obs: dict,
        info: dict,
        previous_info: dict | None = None,
    ) -> None:
        previous_info = previous_info or {}
        action_id = int(action)
        invalid_delta = int(info.get("invalid_actions", 0) or 0) - int(
            previous_info.get("invalid_actions", 0) or 0
        )
        if invalid_delta > 0:
            self.failed_actions.append(
                {
                    "step": obs.get("step"),
                    "action_id": action_id,
                    "action": _action_name(action_id),
                    "agent_pos": obs.get("agent_pos"),
                    "reward": reward,
                }
            )

        if _metric_increased(info, previous_info, "num_build_road"):
            agent_pos = _optional_position(obs.get("agent_pos"))
            if agent_pos is not None:
                self.built_roads.add(agent_pos)

        if _metric_increased(info, previous_info, "num_dig"):
            for pos, tile, _terrain_band in _iter_observed_tiles(obs):
                if self.seen_tiles.get(pos) == int(Tile.OBSTACLE) and tile == int(Tile.GROUND):
                    self.dug_cells.add(pos)

        if action_id == int(Action.MINE) and bool(obs.get("has_ore")) and self.first_mine_step is None:
            self.first_mine_step = _optional_int(obs.get("step"))

        self._merge_road_credit_records(info.get("road_credit_records", []))
        self.update_from_observation(obs)

    def add_reflection(self, reflection: dict) -> None:
        if reflection:
            self.reflections.append(reflection)

    def add_road_credit_records(self, records: Iterable[dict]) -> None:
        self._merge_road_credit_records(records)
        self._episode_index += 1

    def summary(self, max_items: int = 12) -> dict:
        return {
            "known_tile_count": len(self.seen_tiles),
            "known_ore_count": len(self.seen_ore_locations),
            "known_obstacle_count": len(self.seen_obstacles),
            "known_rough_count": len(self.seen_rough_tiles),
            "known_road_count": len(self.seen_roads),
            "known_terrain_band_count": len(self.seen_terrain_bands),
            "visited_cell_count": len(self.visited_counts),
            "memory_updates": self.update_count,
            "known_ore_locations": _positions(self.seen_ore_locations, max_items),
            "first_ore_seen_step": self.first_ore_seen_step,
            "first_mine_step": self.first_mine_step,
            "recent_failed_actions": self.failed_actions[-max_items:],
            "recent_reflections": self.reflections[-3:],
            "road_credit_record_count": len(self.road_credit_records),
            "road_learning_summary": RoadLearningModule.from_records(self.road_credit_records).summary(),
            "least_visited_visible_hint": (
                "Prefer legal moves that reveal new cells or reduce repeated visits when no task cue is visible."
            ),
        }

    def to_dict(self) -> dict:
        return {
            "seen_tiles": [
                {"pos": list(pos), "tile": tile, "tile_name": Tile(tile).name}
                for pos, tile in sorted(self.seen_tiles.items())
            ],
            "seen_ore_locations": _positions(self.seen_ore_locations),
            "seen_obstacles": _positions(self.seen_obstacles),
            "seen_rough_tiles": _positions(self.seen_rough_tiles),
            "seen_roads": _positions(self.seen_roads),
            "seen_terrain_bands": [
                {"pos": list(pos), "terrain_band": band}
                for pos, band in sorted(self.seen_terrain_bands.items())
            ],
            "visited_counts": [
                {"pos": list(pos), "count": count} for pos, count in sorted(self.visited_counts.items())
            ],
            "built_roads": _positions(self.built_roads),
            "dug_cells": _positions(self.dug_cells),
            "road_credit_records": list(self.road_credit_records),
            "failed_actions": list(self.failed_actions),
            "reflections": list(self.reflections),
            "update_count": self.update_count,
            "first_ore_seen_step": self.first_ore_seen_step,
            "first_mine_step": self.first_mine_step,
        }

    def _sync_tile_sets(self, pos: Position, tile: int) -> None:
        self.seen_ore_locations.discard(pos)
        self.seen_obstacles.discard(pos)
        self.seen_rough_tiles.discard(pos)
        self.seen_roads.discard(pos)
        if tile == int(Tile.ORE):
            self.seen_ore_locations.add(pos)
        elif tile == int(Tile.OBSTACLE):
            self.seen_obstacles.add(pos)
        elif tile == int(Tile.ROUGH):
            self.seen_rough_tiles.add(pos)
        elif tile == int(Tile.ROAD):
            self.seen_roads.add(pos)

    def _merge_road_credit_records(self, records: Iterable[dict]) -> None:
        for record in records:
            pos = _optional_position(record.get("position"))
            build_step = _optional_int(record.get("build_step"))
            if pos is None or build_step is None:
                continue
            key = (self._episode_index, pos, build_step)
            clean = dict(record)
            clean["memory_episode"] = self._episode_index
            if key in self._road_credit_index:
                self.road_credit_records[self._road_credit_index[key]] = clean
                continue
            self._road_credit_index[key] = len(self.road_credit_records)
            self.road_credit_records.append(clean)


def _iter_observed_tiles(obs: dict) -> Iterable[tuple[Position, int, str | None]]:
    if obs.get("visible_tiles"):
        for item in obs["visible_tiles"]:
            yield _position(item["pos"]), int(item["tile"]), item.get("terrain_band")
        return

    if "local_view" in obs:
        origin_row, origin_col = obs.get("local_view_origin", [0, 0])
        terrain_bands = obs.get("local_terrain_bands")
        for row_idx, row in enumerate(obs["local_view"]):
            for col_idx, value in enumerate(row):
                if value is not None:
                    band = None
                    if terrain_bands is not None:
                        band = terrain_bands[row_idx][col_idx]
                    yield (origin_row + row_idx, origin_col + col_idx), int(value), band
        return

    if "grid" in obs:
        for row_idx, row in enumerate(obs["grid"]):
            for col_idx, value in enumerate(row):
                yield (row_idx, col_idx), int(value), None


def _position(value) -> Position:
    return int(value[0]), int(value[1])


def _optional_position(value) -> Position | None:
    if value is None:
        return None
    return _position(value)


def _optional_int(value) -> int | None:
    if value is None:
        return None
    return int(value)


def _positions(values: Iterable[Position], limit: int | None = None) -> list[list[int]]:
    ordered = [list(pos) for pos in sorted(values)]
    if limit is not None:
        return ordered[:limit]
    return ordered


def _metric_increased(info: dict, previous_info: dict, key: str) -> bool:
    return int(info.get(key, 0) or 0) > int(previous_info.get(key, 0) or 0)


def _action_name(action_id: int) -> str:
    try:
        return Action(action_id).name
    except ValueError:
        return str(action_id)
