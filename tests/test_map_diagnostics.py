from __future__ import annotations

import unittest

import numpy as np

from evogrid.envs.map_generation.diagnostics import compute_map_diagnostics
from evogrid.envs.map_generation.schemas import MapGenerationConfig


class MapDiagnosticsTest(unittest.TestCase):
    def test_components_spans_and_reachability(self):
        mask = np.array(
            [
                [1, 1, 1, 1, 1],
                [0, 0, 1, 0, 0],
                [1, 1, 1, 1, 1],
                [0, 0, 1, 0, 0],
                [1, 1, 1, 1, 1],
            ],
            dtype=bool,
        )
        roughness = np.zeros((5, 5), dtype=float)

        diagnostics = compute_map_diagnostics(
            mask,
            roughness,
            base_pos=(0, 0),
            ore_positions={(4, 4)},
            config=_config(),
            map_id="manual",
        )

        self.assertEqual(diagnostics["component_count"], 1)
        self.assertEqual(diagnostics["largest_component_size"], 17)
        self.assertTrue(diagnostics["spans_horizontal"])
        self.assertTrue(diagnostics["spans_vertical"])
        self.assertEqual(diagnostics["base_ore_reachable_fraction"], 1.0)
        self.assertEqual(diagnostics["shortest_path_length"], 8)
        self.assertEqual(diagnostics["path_tortuosity"], 1.0)

    def test_no_path_returns_null_path_metrics(self):
        mask = np.zeros((5, 5), dtype=bool)
        mask[0, 0] = True
        mask[4, 4] = True
        diagnostics = compute_map_diagnostics(mask, np.zeros((5, 5)), (0, 0), {(4, 4)}, _config())

        self.assertEqual(diagnostics["component_count"], 2)
        self.assertEqual(diagnostics["base_ore_reachable_fraction"], 0.0)
        self.assertIsNone(diagnostics["shortest_path_length"])
        self.assertIsNone(diagnostics["minimum_cost_path_cost"])
        self.assertIsNone(diagnostics["path_tortuosity"])

    def test_base_equals_goal_has_zero_path(self):
        mask = np.ones((5, 5), dtype=bool)
        diagnostics = compute_map_diagnostics(mask, np.zeros((5, 5)), (2, 2), {(2, 2)}, _config())

        self.assertEqual(diagnostics["shortest_path_length"], 0)
        self.assertEqual(diagnostics["minimum_cost_path_cost"], 0.0)
        self.assertEqual(diagnostics["path_tortuosity"], 0.0)

    def test_minimum_cost_path_uses_roughness(self):
        mask = np.array(
            [
                [1, 1, 1],
                [1, 1, 1],
                [0, 0, 0],
            ],
            dtype=bool,
        )
        roughness = np.array(
            [
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
            ]
        )

        diagnostics = compute_map_diagnostics(mask, roughness, (0, 0), {(0, 2)}, _config())

        self.assertEqual(diagnostics["shortest_path_length"], 2)
        self.assertAlmostEqual(diagnostics["minimum_cost_path_cost"], 0.04)

    def test_articulation_points_for_corridor_and_double_channel(self):
        corridor = np.array([[1, 1, 1, 1, 1]], dtype=bool)
        corridor_diag = compute_map_diagnostics(corridor, np.zeros((1, 5)), (0, 0), {(0, 4)}, _config())
        self.assertEqual(corridor_diag["articulation_point_count"], 3)

        double_channel = np.array(
            [
                [1, 1, 1],
                [1, 0, 1],
                [1, 1, 1],
            ],
            dtype=bool,
        )
        double_diag = compute_map_diagnostics(
            double_channel,
            np.zeros((3, 3)),
            (0, 0),
            {(2, 2)},
            _config(),
        )
        self.assertEqual(double_diag["articulation_point_count"], 0)

    def test_rough_patch_metrics_and_hurst_estimates(self):
        mask = np.ones((8, 8), dtype=bool)
        mask[7, 0] = False
        roughness = np.zeros((8, 8), dtype=float)
        roughness[0:2, 0:2] = 0.9
        roughness[6, 6] = 0.95

        diagnostics = compute_map_diagnostics(mask, roughness, (0, 0), {(7, 7)}, _config())

        self.assertEqual(diagnostics["rough_patch_count"], 2)
        self.assertEqual(diagnostics["largest_rough_patch_fraction"], 4 / 64)
        self.assertIsNotNone(diagnostics["estimated_terrain_hurst"])
        self.assertIsNotNone(diagnostics["estimated_topology_hurst"])

    def test_axis_neighbor_statistics_are_reported(self):
        mask = np.array(
            [
                [1, 1, 0, 0],
                [1, 1, 0, 0],
                [0, 0, 1, 1],
                [0, 0, 1, 1],
            ],
            dtype=bool,
        )
        roughness = np.array(
            [
                [0.0, 0.1, 0.8, 0.9],
                [0.1, 0.2, 0.7, 0.8],
                [0.8, 0.7, 0.2, 0.1],
                [0.9, 0.8, 0.1, 0.0],
            ],
            dtype=float,
        )

        diagnostics = compute_map_diagnostics(mask, roughness, (0, 0), {(1, 1)}, _config())

        for key in (
            "terrain_neighbor_correlation",
            "terrain_axis_corr_abs_diff",
            "terrain_lag1_semivariance_abs_diff",
            "topology_neighbor_correlation",
            "topology_axis_corr_abs_diff",
            "topology_lag1_semivariance_abs_diff",
        ):
            self.assertIn(key, diagnostics)
            self.assertIsNotNone(diagnostics[key])

    def test_diagnostics_do_not_modify_inputs(self):
        mask = np.ones((5, 5), dtype=bool)
        roughness = np.arange(25, dtype=float).reshape(5, 5) / 24.0
        mask_before = mask.copy()
        rough_before = roughness.copy()

        compute_map_diagnostics(mask, roughness, (0, 0), {(4, 4)}, _config())

        np.testing.assert_array_equal(mask, mask_before)
        np.testing.assert_array_equal(roughness, rough_before)

    def test_legacy_carved_is_not_valid_for_percolation_analysis(self):
        diagnostics = compute_map_diagnostics(
            np.ones((5, 5), dtype=bool),
            np.zeros((5, 5)),
            (0, 0),
            {(4, 4)},
            _config(solvability_mode="legacy_carved"),
        )

        self.assertFalse(diagnostics["valid_for_percolation_analysis"])

    def test_rejects_shape_mismatch(self):
        with self.assertRaisesRegex(ValueError, "roughness"):
            compute_map_diagnostics(np.ones((5, 5), dtype=bool), np.zeros((4, 5)))


def _config(solvability_mode: str = "raw") -> MapGenerationConfig:
    return MapGenerationConfig.from_config(
        {
            "env": {
                "map_mode": "fractal_percolation",
                "grid_size": [8, 8],
                "world": {
                    "topology": {
                        "model": "iid_site",
                        "p_open": 0.8,
                        "solvability_mode": solvability_mode,
                    },
                    "terrain": {
                        "base_move_cost": 0.01,
                        "roughness_strength": 0.04,
                        "cost_exponent": 1.0,
                        "observation_bins": [0.25, 0.5, 0.75],
                    },
                    "resources": {"distribution": "uniform", "count": 1},
                },
            }
        }
    )


if __name__ == "__main__":
    unittest.main()
