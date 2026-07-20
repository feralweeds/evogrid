from __future__ import annotations

import json
import unittest

from evogrid.envs.map_builder import build_fixed_map, build_map
from evogrid.envs.map_generation.schemas import MapGenerationConfig


class MapGenerationSchemaTest(unittest.TestCase):
    def test_build_map_wraps_legacy_fixed_map(self):
        result = build_map(seed=7)
        grid, base_pos, ore_positions = build_fixed_map(seed=7)

        self.assertEqual(result.grid, grid)
        self.assertEqual(result.base_pos, base_pos)
        self.assertEqual(result.ore_positions, ore_positions)
        self.assertIsNone(result.roughness)
        self.assertEqual(result.diagnostics["placement_status"], "legacy_carved")
        self.assertFalse(result.diagnostics["valid_for_percolation_analysis"])
        json.dumps(result.to_dict(), sort_keys=True)

    def test_build_fixed_map_keeps_returning_triple(self):
        built = build_fixed_map()

        self.assertIsInstance(built, tuple)
        self.assertEqual(len(built), 3)

    def test_reset_seed_overrides_config_world_seed(self):
        config = {
            "env": {
                "map_mode": "fixed",
                "grid_size": [32, 32],
                "world": {"world_seed": 99},
            }
        }

        parsed = MapGenerationConfig.from_config(config, seed=5)

        self.assertEqual(parsed.world.world_seed, 5)

    def test_unknown_map_mode_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "env.map_mode"):
            build_map({"env": {"map_mode": "surprise"}})

    def test_grid_size_validation_reports_path(self):
        with self.assertRaisesRegex(ValueError, "env.grid_size"):
            MapGenerationConfig.from_config(
                {
                    "env": {
                        "map_mode": "fractal_percolation",
                        "grid_size": [4, 32],
                    }
                }
            )

    def test_legacy_small_maps_remain_allowed(self):
        parsed = MapGenerationConfig.from_config({"env": {"grid_size": [5, 5]}})

        self.assertEqual(parsed.grid_size, (5, 5))

    def test_fractal_config_validation_reports_nested_paths(self):
        config = {
            "env": {
                "map_mode": "fractal_percolation",
                "grid_size": [64, 64],
                "world": {
                    "topology": {
                        "model": "correlated_site",
                        "p_open": 1.2,
                        "hurst": 0.5,
                    }
                },
            }
        }

        with self.assertRaisesRegex(ValueError, "env.world.topology.p_open"):
            MapGenerationConfig.from_config(config)

    def test_observation_bins_must_be_strictly_increasing(self):
        config = {
            "env": {
                "map_mode": "fractal_percolation",
                "grid_size": [64, 64],
                "world": {"terrain": {"observation_bins": [0.25, 0.25, 0.75]}},
            }
        }

        with self.assertRaisesRegex(ValueError, "env.world.terrain.observation_bins"):
            MapGenerationConfig.from_config(config)

    def test_fractal_mode_builds_map_result(self):
        config = {
            "env": {
                "map_mode": "fractal_percolation",
                "grid_size": [64, 64],
                "world": {
                    "topology": {"model": "correlated_site", "p_open": 0.65, "hurst": 0.5},
                    "terrain": {"hurst": 0.7},
                    "resources": {"distribution": "uniform", "count": 1},
                },
            }
        }

        result = build_map(config, seed=0)

        self.assertEqual(result.schema_version, 1)
        self.assertIsNotNone(result.roughness)
        self.assertEqual(result.diagnostics["placement_status"], "ok")


if __name__ == "__main__":
    unittest.main()
