from __future__ import annotations

from collections import Counter
import unittest

from evogrid.constants import Tile
from evogrid.envs import EvoGridMineEnv
from evogrid.envs.map_builder import build_fixed_map, build_map


class FractalPercolationMapTest(unittest.TestCase):
    def test_build_map_returns_complete_result(self):
        result = build_map(_config(), seed=3)

        self.assertEqual(result.schema_version, 1)
        self.assertEqual(len(result.grid), 32)
        self.assertEqual(len(result.grid[0]), 32)
        self.assertIsNotNone(result.roughness)
        self.assertEqual(len(result.roughness), 32)
        self.assertEqual(Tile(result.grid[result.base_pos[0]][result.base_pos[1]]), Tile.BASE)
        self.assertEqual(len(result.ore_positions), 2)
        for ore_pos in result.ore_positions:
            self.assertEqual(Tile(result.grid[ore_pos[0]][ore_pos[1]]), Tile.ORE)
        self.assertEqual(result.diagnostics["placement_status"], "ok")
        self.assertTrue(result.diagnostics["valid_for_percolation_analysis"])
        self.assertIn("substream_seeds", result.provenance)

    def test_same_seed_is_stable_and_different_seed_changes_map(self):
        first = build_map(_config(), seed=5)
        second = build_map(_config(), seed=5)
        third = build_map(_config(), seed=6)

        self.assertEqual(first.map_id, second.map_id)
        self.assertEqual(first.to_dict(), second.to_dict())
        self.assertNotEqual(first.map_id, third.map_id)

    def test_open_count_matches_correlated_target_with_base_and_ore_projection(self):
        result = build_map(_config(p_open=0.5), seed=2)
        counts = Counter(tile for row in result.grid for tile in row)
        projected_open = len(result.grid) * len(result.grid[0]) - counts[int(Tile.OBSTACLE)]

        self.assertEqual(projected_open, round(0.5 * 32 * 32))
        self.assertAlmostEqual(result.diagnostics["realized_p_open"], 0.5)

    def test_build_fixed_map_legacy_triple_works_for_fractal_mode(self):
        grid, base_pos, ore_positions = build_fixed_map(_config(), seed=4)

        self.assertEqual(Tile(grid[base_pos[0]][base_pos[1]]), Tile.BASE)
        self.assertEqual(len(ore_positions), 2)

    def test_env_reset_supports_partial_observation_without_global_leakage(self):
        env = EvoGridMineEnv(_config())

        obs, info = env.reset(seed=7)

        self.assertEqual(obs["observation_mode"], "partial_obs")
        self.assertNotIn("grid", obs)
        self.assertNotIn("ore_positions", obs)
        self.assertNotIn("ore_positions", info["map_summary"])

    def test_matrix_smoke_for_size_hurst_p_and_resource_modes(self):
        for size in (32, 64):
            for hurst in (0.2, 0.5, 0.8):
                for p_open in (0.5, 0.65, 0.8):
                    for resource_distribution in ("uniform", "clustered"):
                        with self.subTest(
                            size=size,
                            hurst=hurst,
                            p_open=p_open,
                            resource_distribution=resource_distribution,
                        ):
                            result = build_map(
                                _config(
                                    size=size,
                                    topology_hurst=hurst,
                                    terrain_hurst=hurst,
                                    p_open=p_open,
                                    resource_distribution=resource_distribution,
                                ),
                                seed=1,
                            )
                            self.assertEqual(result.diagnostics["placement_status"], "ok")
                            self.assertEqual(len(result.ore_positions), 2)


def _config(
    size: int = 32,
    p_open: float = 0.65,
    topology_hurst: float = 0.5,
    terrain_hurst: float = 0.7,
    resource_distribution: str = "uniform",
) -> dict:
    return {
        "env": {
            "map_mode": "fractal_percolation",
            "grid_size": [size, size],
            "max_steps": 200,
            "world": {
                "schema_version": 1,
                "generator_version": "spectral_fbm_v1",
                "topology": {
                    "model": "correlated_site",
                    "p_open": p_open,
                    "hurst": topology_hurst,
                    "solvability_mode": "conditioned_same_component",
                    "task_component": "largest",
                    "min_task_component_fraction": 0.05,
                },
                "terrain": {
                    "hurst": terrain_hurst,
                    "base_move_cost": 0.01,
                    "roughness_strength": 0.04,
                    "cost_exponent": 1.0,
                    "road_move_cost": 0.0,
                    "observation_bins": [0.25, 0.5, 0.75],
                },
                "resources": {
                    "distribution": resource_distribution,
                    "count": 2,
                    "hurst": 0.7,
                    "min_base_distance": max(4, size // 4),
                    "min_pair_distance": max(2, size // 8),
                },
                "placement": {"base_margin": 2, "max_attempts": 100},
            },
            "observation": {"mode": "partial_obs", "local_view_radius": 4},
            "shaping": {"allow_dig": True, "allow_build_road": True},
        }
    }


if __name__ == "__main__":
    unittest.main()
