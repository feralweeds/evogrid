"""Compact cross-chunk memory for agents and planners."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from evogrid.envs.chunk_world.schemas import ChunkCoord


@dataclass
class ChunkMemory:
    known_chunks: dict[ChunkCoord, dict[str, Any]] = field(default_factory=dict)
    frontier_edges: set[tuple[ChunkCoord, str]] = field(default_factory=set)

    def observe_chunk(self, coord: ChunkCoord | tuple[int, int], summary: dict[str, Any]) -> None:
        coord = ChunkCoord.from_any(coord)
        self.known_chunks[coord] = dict(summary)
        for direction in ("north", "south", "east", "west"):
            if summary.get(f"{direction}_open", False):
                self.frontier_edges.add((coord, direction))

    def hierarchical_plan_hint(self, start: tuple[int, int], goal: tuple[int, int], chunk_size: int) -> dict[str, Any]:
        start_chunk = ChunkCoord(start[0] // chunk_size, start[1] // chunk_size)
        goal_chunk = ChunkCoord(goal[0] // chunk_size, goal[1] // chunk_size)
        return {
            "start_chunk": start_chunk.to_tuple(),
            "goal_chunk": goal_chunk.to_tuple(),
            "chunk_delta": (goal_chunk.cx - start_chunk.cx, goal_chunk.cy - start_chunk.cy),
            "known_chunk_count": len(self.known_chunks),
            "frontier_count": len(self.frontier_edges),
        }
