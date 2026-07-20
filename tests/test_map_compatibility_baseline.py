from __future__ import annotations

from collections import Counter, deque
import hashlib
import json
from pathlib import Path
import unittest

import yaml

from evogrid.constants import Tile
from evogrid.envs.map_builder import build_fixed_map


CONFIGS = {
    "fixed": "configs/env_fixed_map.yaml",
    "random": "configs/env_random_curriculum.yaml",
    "controlled": "configs/env_controlled_corridor_curriculum.yaml",
}

EXPECTED_BASELINES = {
    ("fixed", 0): {
        "hash": "066d5c4c0d6a32662ec62c2507b016c0950ffba37c0c4c2919114d5c8fc05839",
        "base_pos": [2, 2],
        "ore_positions": [[26, 26]],
        "tile_counts": {"BASE": 1, "GROUND": 813, "OBSTACLE": 28, "ORE": 1, "ROUGH": 181},
    },
    ("fixed", 1): {
        "hash": "066d5c4c0d6a32662ec62c2507b016c0950ffba37c0c4c2919114d5c8fc05839",
        "base_pos": [2, 2],
        "ore_positions": [[26, 26]],
        "tile_counts": {"BASE": 1, "GROUND": 813, "OBSTACLE": 28, "ORE": 1, "ROUGH": 181},
    },
    ("fixed", 2): {
        "hash": "066d5c4c0d6a32662ec62c2507b016c0950ffba37c0c4c2919114d5c8fc05839",
        "base_pos": [2, 2],
        "ore_positions": [[26, 26]],
        "tile_counts": {"BASE": 1, "GROUND": 813, "OBSTACLE": 28, "ORE": 1, "ROUGH": 181},
    },
    ("random", 0): {
        "hash": "f7a0395ff769862bb2f8e4a8536b6993a969caffcee79e2a246fc150ca75bc06",
        "base_pos": [2, 2],
        "ore_positions": [[11, 15]],
        "tile_counts": {"BASE": 1, "GROUND": 161, "OBSTACLE": 30, "ORE": 1, "ROUGH": 63},
    },
    ("random", 1): {
        "hash": "c5b3bd0bb44ff632765e218d65d62b25f4b5a0e498b9bf118fd08b6c2e6f7a46",
        "base_pos": [2, 2],
        "ore_positions": [[6, 9]],
        "tile_counts": {"BASE": 1, "GROUND": 166, "OBSTACLE": 27, "ORE": 1, "ROUGH": 61},
    },
    ("random", 2): {
        "hash": "86ae8756c6dbbe4a661a1b42cac38684e54967f6a8c66008933377d0c5d6f0ef",
        "base_pos": [2, 2],
        "ore_positions": [[2, 15]],
        "tile_counts": {"BASE": 1, "GROUND": 164, "OBSTACLE": 34, "ORE": 1, "ROUGH": 56},
    },
    ("controlled", 0): {
        "hash": "145b48089211b540efb49cfb8d52907ec95c74c246cd5b5bb50f878db7765c2b",
        "base_pos": [10, 13],
        "ore_positions": [[0, 10]],
        "tile_counts": {"BASE": 1, "GROUND": 215, "OBSTACLE": 17, "ORE": 1, "ROUGH": 22},
    },
    ("controlled", 1): {
        "hash": "c79be97480cd683448a11bdc83f836e553d0f6d8339c635514d251c8db650df4",
        "base_pos": [3, 6],
        "ore_positions": [[11, 4]],
        "tile_counts": {"BASE": 1, "GROUND": 195, "OBSTACLE": 23, "ORE": 1, "ROUGH": 36},
    },
    ("controlled", 2): {
        "hash": "391daaf17fbaad3a033dc8337dec571dcfbc840bcb484b28ec37b32dba4d4f18",
        "base_pos": [3, 4],
        "ore_positions": [[5, 12]],
        "tile_counts": {"BASE": 1, "GROUND": 224, "OBSTACLE": 21, "ORE": 1, "ROUGH": 9},
    },
}


class MapCompatibilityBaselineTest(unittest.TestCase):
    def test_legacy_map_modes_match_frozen_baseline(self):
        for key, config_path in CONFIGS.items():
            config = _load_yaml(config_path)
            for seed in (0, 1, 2):
                with self.subTest(map_mode=key, seed=seed):
                    actual = _map_summary(config, seed)
                    expected = EXPECTED_BASELINES[(key, seed)]
                    self.assertEqual(actual["hash"], expected["hash"])
                    self.assertEqual(actual["base_pos"], expected["base_pos"])
                    self.assertEqual(actual["ore_positions"], expected["ore_positions"])
                    self.assertEqual(actual["tile_counts"], expected["tile_counts"])
                    self.assertTrue(actual["all_ore_reachable"])


def _load_yaml(path: str) -> dict:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def _map_summary(config: dict, seed: int) -> dict:
    grid, base_pos, ore_positions = build_fixed_map(config, seed=seed)
    payload = {
        "grid": grid,
        "base_pos": list(base_pos),
        "ore_positions": [list(pos) for pos in sorted(ore_positions)],
    }
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    counts = Counter(tile for row in grid for tile in row)
    return {
        "hash": digest,
        "base_pos": list(base_pos),
        "ore_positions": [list(pos) for pos in sorted(ore_positions)],
        "tile_counts": {Tile(tile).name: counts[tile] for tile in sorted(counts)},
        "all_ore_reachable": all(_reachable(grid, base_pos, ore_pos) for ore_pos in ore_positions),
    }


def _reachable(grid: list[list[int]], start: tuple[int, int], goal: tuple[int, int]) -> bool:
    queue = deque([start])
    seen = {start}
    while queue:
        row, col = queue.popleft()
        if (row, col) == goal:
            return True
        for next_pos in ((row - 1, col), (row + 1, col), (row, col - 1), (row, col + 1)):
            if next_pos in seen:
                continue
            next_row, next_col = next_pos
            if not (0 <= next_row < len(grid) and 0 <= next_col < len(grid[0])):
                continue
            if Tile(grid[next_row][next_col]) == Tile.OBSTACLE:
                continue
            seen.add(next_pos)
            queue.append(next_pos)
    return False


if __name__ == "__main__":
    unittest.main()
