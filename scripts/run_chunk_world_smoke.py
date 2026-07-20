from __future__ import annotations

try:
    import _bootstrap  # type: ignore  # noqa: F401
except ModuleNotFoundError:
    from scripts import _bootstrap  # noqa: F401

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from evogrid.constants import Tile
from evogrid.envs.chunk_world import ChunkConfig, ChunkEventStore, ChunkMemory, ChunkWorld
from evogrid.skills.context import SkillContext
from evogrid.skills.runtime import SkillRuntime
from evogrid.skills.schemas import SkillSpec


def run_chunk_world_smoke(config_path: str | Path, out_dir: str | Path) -> dict[str, Any]:
    config_path = Path(config_path)
    out_dir = Path(out_dir)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    chunk_config = ChunkConfig.from_dict(config.get("chunk_world", config))
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "maps").mkdir(exist_ok=True)
    (out_dir / "episodes").mkdir(exist_ok=True)
    (out_dir / "skills").mkdir(exist_ok=True)
    (out_dir / "capability").mkdir(exist_ok=True)

    event_path = out_dir / "chunk_events.jsonl"
    world = ChunkWorld(chunk_config, ChunkEventStore(event_path))
    coords = [(0, 0), (1, 0), (0, 1), (-1, 0)]
    forward_checksums = [_chunk_checksum(world.get_chunk(coord)) for coord in coords]
    reverse_world = ChunkWorld(chunk_config, ChunkEventStore())
    reverse_checksums = [_chunk_checksum(reverse_world.get_chunk(coord)) for coord in reversed(coords)]
    order_independent = forward_checksums == list(reversed(reverse_checksums))
    boundary = _boundary_metrics(world)

    event_x, event_y = chunk_config.chunk_size + 1, 2
    world.apply_event({"event_type": "dig", "x": event_x, "y": event_y, "actor_id": "chunk_smoke"})
    world.apply_event({"event_type": "build_road", "x": event_x, "y": event_y, "actor_id": "chunk_smoke"})
    world.apply_event({"event_type": "deplete_ore", "x": event_x, "y": event_y, "actor_id": "chunk_smoke"})
    world.unload_chunk((1, 0))
    reloaded = ChunkWorld(chunk_config, ChunkEventStore(event_path))
    event_survived = bool(
        reloaded.local_observation((event_x, event_y), 0)["roads"][0, 0]
        and reloaded.local_observation((event_x, event_y), 0)["walkable"][0, 0]
        and reloaded.local_observation((event_x, event_y), 0)["depleted"][0, 0]
    )

    transfer = _skill_transfer_smoke(reloaded, chunk_config, config.get("skill_transfer", {}))
    memory = ChunkMemory()
    for coord in coords[:2]:
        memory.observe_chunk(coord, reloaded.chunk_summary(coord))
    memory_hint = memory.hierarchical_plan_hint((0, 0), (chunk_config.chunk_size * 2 + 1, 0), chunk_config.chunk_size)

    chunk_rows = [
        {
            "schema_version": 1,
            "coord": f"{coord[0]},{coord[1]}",
            "checksum": checksum,
            "order": "forward",
        }
        for coord, checksum in zip(coords, forward_checksums)
    ]
    _write_jsonl(out_dir / "maps" / "chunk_manifest.jsonl", chunk_rows)
    _write_csv(
        out_dir / "capability" / "chunk_world_metrics.csv",
        [
            {
                "schema_version": 1,
                "order_independent": order_independent,
                "east_west_halo_max_abs_diff": boundary["east_west_halo_max_abs_diff"],
                "north_south_halo_max_abs_diff": boundary["north_south_halo_max_abs_diff"],
                "event_survived_reload": event_survived,
                "skill_transfer_completed": transfer["completed"],
                "cache_count_after_smoke": reloaded.cached_chunk_count(),
            }
        ],
    )
    _write_jsonl(out_dir / "episodes" / "memory_trace.jsonl", [{"schema_version": 1, **memory_hint}])
    _write_jsonl(out_dir / "skills" / "skill_transfer_trace.jsonl", [transfer])

    manifest = {
        "schema_version": 1,
        "run_id": out_dir.name,
        "experiment_type": "chunk_world_smoke",
        "mode": str(config.get("experiment", {}).get("mode", "mock_smoke")),
        "started_at": datetime.now(timezone.utc).isoformat(),
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "config": str(config_path),
        "chunk_config": chunk_config.to_dict(),
        "order_independent": order_independent,
        "boundary_metrics": boundary,
        "event_survived_reload": event_survived,
        "cache_bound": chunk_config.max_cached_chunks,
        "cache_count_after_smoke": reloaded.cached_chunk_count(),
        "skill_transfer": transfer,
        "memory_hint": memory_hint,
        "completion_status": "completed",
    }
    (out_dir / "run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return manifest


def _boundary_metrics(world: ChunkWorld) -> dict[str, float]:
    config = world.config
    left = world.get_chunk((0, 0))
    right = world.get_chunk((1, 0))
    north = world.get_chunk((0, -1))
    south = world.get_chunk((0, 0))
    rows = slice(config.halo, config.halo + config.chunk_size)
    cols = slice(config.halo, config.halo + config.chunk_size)
    east_west = np.max(
        np.abs(
            left.roughness[rows, config.halo + config.chunk_size]
            - right.roughness[rows, config.halo]
        )
    )
    north_south = np.max(
        np.abs(
            north.roughness[config.halo + config.chunk_size, cols]
            - south.roughness[config.halo, cols]
        )
    )
    return {
        "east_west_halo_max_abs_diff": float(east_west),
        "north_south_halo_max_abs_diff": float(north_south),
    }


def _skill_transfer_smoke(world: ChunkWorld, config: ChunkConfig, transfer_config: dict[str, Any]) -> dict[str, Any]:
    skill = _transfer_skill(transfer_config)
    obs = _chunk_skill_observation(world, (config.chunk_size + 1, 2), radius=1)
    context = SkillContext.from_observable_inputs(
        observation=obs,
        info={"world_mode": "chunk", "visible_radius": 1},
        memory_summary={"visit_count_bucket": 1, "similar_outcome_count": 0, "similar_mean_payoff": 0.0},
        route_plan={"exists": True, "is_known_transport_route": False, "remaining_length_bucket": 2},
        episode_budget={"steps_remaining": 20},
    )
    result = SkillRuntime().execute(skill, context, run_id="chunk_world_smoke", episode_id="transfer/0", step=0)
    return {
        "skill_id": skill.skill_id,
        "schema_version": 1,
        "skill_version": skill.version,
        "spec_hash": skill.spec_hash,
        "completed": result.completed,
        "termination": result.termination,
        "chosen_action": result.chosen_action,
        "trace": result.trace.to_dict(),
    }


def _chunk_skill_observation(world: ChunkWorld, center: tuple[int, int], radius: int) -> dict[str, Any]:
    local = world.local_observation(center, radius)
    visible_tiles = []
    y0 = center[1] - radius
    x0 = center[0] - radius
    for row in range(local["walkable"].shape[0]):
        for col in range(local["walkable"].shape[1]):
            x = x0 + col
            y = y0 + row
            if local["roads"][row, col]:
                tile = int(Tile.ROAD)
            elif local["ore"][row, col]:
                tile = int(Tile.ORE)
            elif not local["walkable"][row, col]:
                tile = int(Tile.OBSTACLE)
            elif local["roughness"][row, col] > 0.66:
                tile = int(Tile.ROUGH)
            else:
                tile = int(Tile.GROUND)
            visible_tiles.append(
                {
                    "pos": [x, y],
                    "tile": tile,
                    "terrain_band": _terrain_band(float(local["roughness"][row, col])),
                }
            )
    return {"agent_pos": list(center), "visible_tiles": visible_tiles, "has_ore": False, "step": 0}


def _transfer_skill(config: dict[str, Any]) -> SkillSpec:
    return SkillSpec.from_dict(
        {
            "schema_version": 1,
            "skill_id": str(config.get("skill_id", "finite_map_route_bias")),
            "version": str(config.get("version", "1.0.0")),
            "status": "verified",
            "name": "Finite Map Route Bias",
            "description": "Fixture skill transferred from finite-map route shaping to chunk-local observation.",
            "problem_addressed": "Choose a route action using only local chunk observations and memory summaries.",
            "source": {
                "proposer": "fixture",
                "source_partition": "verify",
                "source_episode_ids": ["finite_map/verified/0"],
            },
            "applicability": {
                "all": [
                    {"feature": "route.exists", "op": "eq", "value": True},
                    {"feature": "current.tile_type", "op": "in", "value": [0, 4, 5]},
                ]
            },
            "procedure": [{"op": "ACT", "action": "MOVE_RIGHT"}],
            "budget": {
                "max_runtime_steps": 3,
                "max_environment_actions": 1,
                "max_nested_skill_depth": 0,
            },
            "objective": {"primary_metric": "chunk_transfer_success", "direction": "maximize"},
        }
    )


def _terrain_band(value: float) -> str:
    if value < 0.33:
        return "low"
    if value < 0.66:
        return "medium"
    return "high"


def _chunk_checksum(chunk) -> str:
    digest = hashlib.sha256()
    for array in (chunk.walkable, chunk.roughness, chunk.ore, chunk.roads, chunk.depleted):
        digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run M7 chunk-world smoke checks.")
    parser.add_argument("--config", default="configs/chunk_world_smoke.yaml")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    manifest = run_chunk_world_smoke(args.config, args.out)
    print(json.dumps({"run_manifest": str(Path(args.out) / "run_manifest.json"), "status": manifest["completion_status"]}))


if __name__ == "__main__":
    main()
