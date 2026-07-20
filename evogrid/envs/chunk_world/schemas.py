"""Schemas for chunked open-world generation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True, order=True)
class ChunkCoord:
    cx: int
    cy: int

    @classmethod
    def from_any(cls, value: "ChunkCoord | tuple[int, int] | list[int]") -> "ChunkCoord":
        if isinstance(value, ChunkCoord):
            return value
        if len(value) != 2:
            raise ValueError("chunk coord: expected (cx, cy)")
        return cls(int(value[0]), int(value[1]))

    def to_tuple(self) -> tuple[int, int]:
        return (self.cx, self.cy)

    def to_key(self) -> str:
        return f"{self.cx},{self.cy}"


@dataclass(frozen=True)
class ChunkConfig:
    root_seed: int
    chunk_size: int = 16
    halo: int = 1
    p_open: float = 0.68
    topology_hurst: float = 0.55
    terrain_hurst: float = 0.45
    ore_threshold: float = 0.92
    max_cached_chunks: int = 32

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ChunkConfig":
        config = cls(
            root_seed=int(data.get("root_seed", 0)),
            chunk_size=int(data.get("chunk_size", 16)),
            halo=int(data.get("halo", 1)),
            p_open=float(data.get("p_open", 0.68)),
            topology_hurst=float(data.get("topology_hurst", 0.55)),
            terrain_hurst=float(data.get("terrain_hurst", 0.45)),
            ore_threshold=float(data.get("ore_threshold", 0.92)),
            max_cached_chunks=int(data.get("max_cached_chunks", 32)),
        )
        config.validate()
        return config

    def validate(self) -> None:
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if self.halo < 0:
            raise ValueError("halo must be non-negative")
        if not (0.0 < self.p_open < 1.0):
            raise ValueError("p_open must satisfy 0 < p_open < 1")
        for name, value in (("topology_hurst", self.topology_hurst), ("terrain_hurst", self.terrain_hurst)):
            if not (0.0 < value < 1.0):
                raise ValueError(f"{name} must satisfy 0 < H < 1")
        if not (0.0 < self.ore_threshold < 1.0):
            raise ValueError("ore_threshold must satisfy 0 < ore_threshold < 1")
        if self.max_cached_chunks <= 0:
            raise ValueError("max_cached_chunks must be positive")

    def to_dict(self) -> dict[str, Any]:
        return {
            "root_seed": self.root_seed,
            "chunk_size": self.chunk_size,
            "halo": self.halo,
            "p_open": self.p_open,
            "topology_hurst": self.topology_hurst,
            "terrain_hurst": self.terrain_hurst,
            "ore_threshold": self.ore_threshold,
            "max_cached_chunks": self.max_cached_chunks,
        }


@dataclass(frozen=True)
class ChunkSnapshot:
    coord: ChunkCoord
    walkable: np.ndarray
    roughness: np.ndarray
    ore: np.ndarray
    roads: np.ndarray
    depleted: np.ndarray
    halo: int

    @property
    def shape(self) -> tuple[int, int]:
        return tuple(int(item) for item in self.walkable.shape)

    def interior_slice(self) -> tuple[slice, slice]:
        if self.halo == 0:
            return (slice(None), slice(None))
        return (slice(self.halo, -self.halo), slice(self.halo, -self.halo))

    def interior(self) -> "ChunkSnapshot":
        yy, xx = self.interior_slice()
        return ChunkSnapshot(
            coord=self.coord,
            walkable=self.walkable[yy, xx].copy(),
            roughness=self.roughness[yy, xx].copy(),
            ore=self.ore[yy, xx].copy(),
            roads=self.roads[yy, xx].copy(),
            depleted=self.depleted[yy, xx].copy(),
            halo=0,
        )
