from __future__ import annotations

from collections import deque
import unittest

from evogrid.constants import Tile
from evogrid.envs import EvoGridMineEnv
from evogrid.envs.map_builder import build_fixed_map


class RandomMapBuilderTest(unittest.TestCase):
    def test_random_curriculum_is_reproducible_for_same_seed(self):
        config = _random_config()

        first = build_fixed_map(config, seed=123)
        second = build_fixed_map(config, seed=123)

        self.assertEqual(first, second)

    def test_random_curriculum_changes_across_seeds(self):
        config = _random_config()

        first_grid, first_base, first_ore = build_fixed_map(config, seed=123)
        second_grid, second_base, second_ore = build_fixed_map(config, seed=124)

        self.assertEqual(first_base, second_base)
        self.assertNotEqual((first_grid, first_ore), (second_grid, second_ore))

    def test_random_curriculum_keeps_all_seeded_maps_reachable(self):
        config = _random_config()

        for seed in range(50):
            with self.subTest(seed=seed):
                grid, base_pos, ore_positions = build_fixed_map(config, seed=seed)
                self.assertEqual(Tile(grid[base_pos[0]][base_pos[1]]), Tile.BASE)
                self.assertEqual(len(ore_positions), 1)
                for ore_pos in ore_positions:
                    self.assertEqual(Tile(grid[ore_pos[0]][ore_pos[1]]), Tile.ORE)
                    self.assertTrue(_reachable(grid, base_pos, ore_pos))

    def test_random_curriculum_partial_observation_does_not_leak_global_ore_positions(self):
        env = EvoGridMineEnv(_random_config())

        obs, info = env.reset(seed=0)

        self.assertEqual(obs["observation_mode"], "partial_obs")
        self.assertNotIn("grid", obs)
        self.assertNotIn("ore_positions", obs)
        self.assertNotIn("ore_positions", info["map_summary"])

    def test_controlled_corridor_curriculum_is_reproducible_for_same_seed(self):
        config = _controlled_config()

        first = build_fixed_map(config, seed=321)
        second = build_fixed_map(config, seed=321)

        self.assertEqual(first, second)

    def test_controlled_corridor_curriculum_changes_across_seeds(self):
        config = _controlled_config()

        first = build_fixed_map(config, seed=321)
        second = build_fixed_map(config, seed=322)

        self.assertNotEqual(first, second)

    def test_controlled_corridor_curriculum_keeps_seeded_maps_reachable(self):
        config = _controlled_config()

        for seed in range(50):
            with self.subTest(seed=seed):
                grid, base_pos, ore_positions = build_fixed_map(config, seed=seed)
                self.assertEqual(Tile(grid[base_pos[0]][base_pos[1]]), Tile.BASE)
                for ore_pos in ore_positions:
                    self.assertEqual(Tile(grid[ore_pos[0]][ore_pos[1]]), Tile.ORE)
                    self.assertTrue(_reachable(grid, base_pos, ore_pos))

    def test_controlled_positive_maps_have_route_rough_signal(self):
        route_rough_counts = []
        for seed in range(30):
            env = EvoGridMineEnv(_controlled_config("positive"))
            obs, info = env.reset(seed=seed)
            self.assertEqual(obs["observation_mode"], "partial_obs")
            route_rough_counts.append(int(info["route_rough_tile_count"]))

        self.assertGreaterEqual(sum(1 for count in route_rough_counts if count > 0), 24)
        self.assertGreater(sum(route_rough_counts) / len(route_rough_counts), 1.5)

    def test_controlled_negative_maps_make_rough_mostly_distracting(self):
        off_route_wins = 0
        for seed in range(30):
            env = EvoGridMineEnv(_controlled_config("negative"))
            obs, info = env.reset(seed=seed)
            self.assertEqual(obs["observation_mode"], "partial_obs")
            if int(info["off_route_rough_tile_count"]) >= int(info["route_rough_tile_count"]):
                off_route_wins += 1

        self.assertGreaterEqual(off_route_wins, 24)


def _random_config() -> dict:
    return {
        "env": {
            "map_mode": "random_curriculum",
            "grid_size": [16, 16],
            "base_pos": [2, 2],
            "random_map": {
                "ore_count": 1,
                "min_base_ore_distance": 10,
                "obstacle_density": 0.12,
                "rough_density": 0.25,
                "rough_corridor_bias": 0.4,
                "ensure_reachable": True,
                "max_generation_attempts": 100,
            },
            "observation": {"mode": "partial_obs", "local_view_radius": 4},
            "shaping": {"allow_dig": True, "allow_build_road": True},
        }
    }


def _controlled_config(scenario: str | None = None) -> dict:
    weights = {
        "positive_weight": 0.60,
        "mixed_weight": 0.20,
        "negative_weight": 0.20,
    }
    if scenario is not None:
        weights = {
            "positive_weight": 1.0 if scenario == "positive" else 0.0,
            "mixed_weight": 1.0 if scenario == "mixed" else 0.0,
            "negative_weight": 1.0 if scenario == "negative" else 0.0,
        }
    return {
        "env": {
            "map_mode": "controlled_corridor_curriculum",
            "grid_size": [16, 16],
            "base_pos": [2, 2],
            "random_map": {
                "ore_count": 1,
                "min_base_ore_distance": 9,
                "obstacle_density": 0.08,
                "extra_rough_density": 0.06,
                "max_generation_attempts": 100,
                "controlled_corridor": {
                    **weights,
                    "base_margin": 2,
                    "positive_route_rough_probability": 0.70,
                    "mixed_route_rough_probability": 0.38,
                    "negative_route_rough_probability": 0.06,
                    "positive_off_route_rough_probability": 0.08,
                    "mixed_off_route_rough_probability": 0.24,
                    "negative_off_route_rough_probability": 0.34,
                    "positive_min_route_rough": 4,
                    "mixed_min_route_rough": 2,
                    "negative_min_route_rough": 0,
                    "positive_transport_band_probability": 0.70,
                    "mixed_transport_band_probability": 0.35,
                    "negative_transport_band_probability": 0.0,
                },
            },
            "observation": {"mode": "partial_obs", "local_view_radius": 4},
            "shaping": {"allow_dig": True, "allow_build_road": True},
        }
    }


def _reachable(grid: list[list[int]], start: tuple[int, int], goal: tuple[int, int]) -> bool:
    queue = deque([start])
    seen = {start}
    while queue:
        row, col = queue.popleft()
        if (row, col) == goal:
            return True
        for nxt in ((row - 1, col), (row + 1, col), (row, col - 1), (row, col + 1)):
            if nxt in seen:
                continue
            next_row, next_col = nxt
            if not (0 <= next_row < len(grid) and 0 <= next_col < len(grid[0])):
                continue
            if Tile(grid[next_row][next_col]) == Tile.OBSTACLE:
                continue
            seen.add(nxt)
            queue.append(nxt)
    return False


if __name__ == "__main__":
    unittest.main()
