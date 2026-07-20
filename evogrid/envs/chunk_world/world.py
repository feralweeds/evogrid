"""Static chunk cache with dynamic event replay."""

from __future__ import annotations

from collections import OrderedDict
from typing import Any

import numpy as np

from evogrid.envs.chunk_world.event_store import ChunkEventStore
from evogrid.envs.chunk_world.global_field import GlobalCoordinateField
from evogrid.envs.chunk_world.schemas import ChunkConfig, ChunkCoord, ChunkSnapshot


class ChunkWorld:
    def __init__(self, config: ChunkConfig | dict[str, Any], event_store: ChunkEventStore | None = None):
        self.config = config if isinstance(config, ChunkConfig) else ChunkConfig.from_dict(config)
        self.config.validate()
        self.field = GlobalCoordinateField(self.config.root_seed)
        self.event_store = event_store or ChunkEventStore()
        self._static_cache: OrderedDict[ChunkCoord, ChunkSnapshot] = OrderedDict()

    def get_chunk(self, coord: ChunkCoord | tuple[int, int]) -> ChunkSnapshot:
        coord = ChunkCoord.from_any(coord)
        static = self._get_static_chunk(coord)
        return self._apply_events(static)

    def unload_chunk(self, coord: ChunkCoord | tuple[int, int]) -> None:
        coord = ChunkCoord.from_any(coord)
        self._static_cache.pop(coord, None)

    def cached_chunk_count(self) -> int:
        return len(self._static_cache)

    def apply_event(self, event: dict[str, Any]) -> None:
        self.event_store.append(event)

    def local_observation(self, center: tuple[int, int], radius: int) -> dict[str, np.ndarray]:
        if radius < 0:
            raise ValueError("radius must be non-negative")
        x0, y0 = int(center[0]), int(center[1])
        xs = range(x0 - radius, x0 + radius + 1)
        ys = range(y0 - radius, y0 + radius + 1)
        topology = self.field.sample_grid(xs, ys, channel="topology", hurst=self.config.topology_hurst)
        roughness = self.field.sample_grid(xs, ys, channel="terrain", hurst=self.config.terrain_hurst)
        ore_field = self.field.sample_grid(xs, ys, channel="ore", hurst=self.config.terrain_hurst)
        walkable = topology < self.config.p_open
        ore = ore_field >= self.config.ore_threshold
        roads = np.zeros_like(walkable, dtype=bool)
        depleted = np.zeros_like(walkable, dtype=bool)
        events = self.event_store.events_for_bounds(x0 - radius, x0 + radius, y0 - radius, y0 + radius)
        for event in events:
            row = event.y - (y0 - radius)
            col = event.x - (x0 - radius)
            if event.event_type == "build_road":
                roads[row, col] = bool(event.value)
            elif event.event_type == "dig":
                walkable[row, col] = bool(event.value)
            elif event.event_type == "deplete_ore":
                depleted[row, col] = bool(event.value)
                if depleted[row, col]:
                    ore[row, col] = False
        return {
            "walkable": walkable,
            "roughness": roughness,
            "ore": ore,
            "roads": roads,
            "depleted": depleted,
        }

    def chunk_summary(self, coord: ChunkCoord | tuple[int, int]) -> dict[str, Any]:
        chunk = self.get_chunk(coord).interior()
        return {
            "coord": chunk.coord.to_tuple(),
            "walkable_ratio": float(chunk.walkable.mean()),
            "ore_count": int(chunk.ore.sum()),
            "road_count": int(chunk.roads.sum()),
            "north_open": bool(chunk.walkable[0, :].any()),
            "south_open": bool(chunk.walkable[-1, :].any()),
            "west_open": bool(chunk.walkable[:, 0].any()),
            "east_open": bool(chunk.walkable[:, -1].any()),
        }

    def _get_static_chunk(self, coord: ChunkCoord) -> ChunkSnapshot:
        if coord in self._static_cache:
            self._static_cache.move_to_end(coord)
            return self._static_cache[coord]
        chunk = self._generate_static_chunk(coord)
        self._static_cache[coord] = chunk
        self._static_cache.move_to_end(coord)
        while len(self._static_cache) > self.config.max_cached_chunks:
            self._static_cache.popitem(last=False)
        return chunk

    def _generate_static_chunk(self, coord: ChunkCoord) -> ChunkSnapshot:
        size = self.config.chunk_size
        halo = self.config.halo
        x_start = coord.cx * size - halo
        y_start = coord.cy * size - halo
        xs = range(x_start, x_start + size + 2 * halo)
        ys = range(y_start, y_start + size + 2 * halo)
        topology = self.field.sample_grid(xs, ys, channel="topology", hurst=self.config.topology_hurst)
        roughness = self.field.sample_grid(xs, ys, channel="terrain", hurst=self.config.terrain_hurst)
        ore_field = self.field.sample_grid(xs, ys, channel="ore", hurst=self.config.terrain_hurst)
        walkable = topology < self.config.p_open
        ore = ore_field >= self.config.ore_threshold
        roads = np.zeros_like(walkable, dtype=bool)
        depleted = np.zeros_like(walkable, dtype=bool)
        return ChunkSnapshot(coord, walkable, roughness, ore, roads, depleted, halo=halo)

    def _apply_events(self, static: ChunkSnapshot) -> ChunkSnapshot:
        size = self.config.chunk_size
        halo = self.config.halo
        x_min = static.coord.cx * size - halo
        x_max = x_min + size + 2 * halo - 1
        y_min = static.coord.cy * size - halo
        y_max = y_min + size + 2 * halo - 1
        walkable = static.walkable.copy()
        roughness = static.roughness.copy()
        ore = static.ore.copy()
        roads = static.roads.copy()
        depleted = static.depleted.copy()
        for event in self.event_store.events_for_bounds(x_min, x_max, y_min, y_max):
            row = event.y - y_min
            col = event.x - x_min
            if event.event_type == "build_road":
                roads[row, col] = bool(event.value)
            elif event.event_type == "dig":
                walkable[row, col] = bool(event.value)
            elif event.event_type == "deplete_ore":
                depleted[row, col] = bool(event.value)
                if depleted[row, col]:
                    ore[row, col] = False
        return ChunkSnapshot(static.coord, walkable, roughness, ore, roads, depleted, halo=halo)
