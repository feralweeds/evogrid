"""Fractal/percolation finite-map generator."""

from __future__ import annotations

from dataclasses import asdict
from typing import Protocol

import numpy as np

from evogrid.constants import Tile
from evogrid.envs.map_generation.connectivity import generate_open_mask
from evogrid.envs.map_generation.diagnostics import compute_map_diagnostics
from evogrid.envs.map_generation.placement import place_base_and_resources
from evogrid.envs.map_generation.schemas import MapBuildResult, MapGenerationConfig, stable_map_id
from evogrid.envs.map_generation.seeding import derive_seed
from evogrid.envs.map_generation.spectral_field import generate_rank_normalized_field


class MapGenerator(Protocol):
    name: str
    version: str

    def build(self, config: MapGenerationConfig, seed: int) -> MapBuildResult:
        ...


class FractalPercolationMapGenerator:
    name = "fractal_percolation"
    version = "spectral_fbm_v1"

    def build(self, config: MapGenerationConfig, seed: int) -> MapBuildResult:
        substream_seeds = {
            "topology": derive_seed(seed, "topology"),
            "terrain": derive_seed(seed, "terrain"),
            "resources": derive_seed(seed, "resources"),
            "placement": derive_seed(seed, "placement"),
        }
        open_mask = generate_open_mask(config, substream_seeds["topology"])
        roughness = generate_rank_normalized_field(
            config.grid_size,
            config.world.terrain.hurst,
            substream_seeds["terrain"],
        )
        placement = place_base_and_resources(
            open_mask,
            config,
            seed=substream_seeds["placement"],
            resource_seed=substream_seeds["resources"],
        )
        if not placement.ok or placement.base_pos is None:
            reason = placement.diagnostics.get("placement_failure_reason", "unknown")
            raise RuntimeError(f"Map placement failed for seed {seed}: {reason}")

        grid = _project_grid(open_mask, placement.base_pos, placement.ore_positions)
        map_id = stable_map_id(
            {
                "schema_version": config.world.schema_version,
                "map_mode": config.map_mode,
                "generator_version": self.version,
                "world_seed": seed,
                "config": asdict(config),
                "open_mask": open_mask.astype(int).tolist(),
                "roughness": roughness.tolist(),
                "base_pos": placement.base_pos,
                "ore_positions": placement.ore_positions,
            }
        )
        diagnostics = compute_map_diagnostics(
            open_mask=open_mask,
            roughness=roughness,
            base_pos=placement.base_pos,
            ore_positions=placement.ore_positions,
            config=config,
            map_id=map_id,
            placement_status=placement.placement_status,
        )
        diagnostics.update(placement.diagnostics)
        provenance = {
            "schema_version": 1,
            "map_mode": config.map_mode,
            "generator": self.name,
            "generator_version": self.version,
            "world_seed": seed,
            "substream_seeds": substream_seeds,
            "config": asdict(config),
        }
        return MapBuildResult(
            schema_version=1,
            map_id=map_id,
            grid=grid,
            roughness=roughness.tolist(),
            base_pos=placement.base_pos,
            ore_positions=placement.ore_positions,
            diagnostics=diagnostics,
            provenance=provenance,
        )


def _project_grid(
    open_mask: np.ndarray,
    base_pos: tuple[int, int],
    ore_positions: set[tuple[int, int]],
) -> list[list[int]]:
    grid = [
        [int(Tile.GROUND) if bool(open_mask[row, col]) else int(Tile.OBSTACLE) for col in range(open_mask.shape[1])]
        for row in range(open_mask.shape[0])
    ]
    base_row, base_col = base_pos
    grid[base_row][base_col] = int(Tile.BASE)
    for ore_row, ore_col in ore_positions:
        grid[ore_row][ore_col] = int(Tile.ORE)
    return grid
