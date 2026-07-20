from __future__ import annotations

import unittest

import numpy as np

from evogrid.envs.map_generation.connectivity import label_components
from evogrid.envs.map_generation.placement import place_base_and_resources
from evogrid.envs.map_generation.schemas import MapGenerationConfig


class MapPlacementTest(unittest.TestCase):
    def test_conditioned_places_base_and_ores_in_same_component(self):
        mask = np.ones((12, 12), dtype=bool)
        mask[6, :] = False
        config = _config(solvability_mode="conditioned_same_component", count=3, min_base_distance=2)

        result = place_base_and_resources(mask, config, seed=10)
        index = label_components(mask)

        self.assertTrue(result.ok)
        self.assertIsNotNone(result.base_pos)
        base_label = int(index.labels[result.base_pos])
        self.assertNotEqual(base_label, 0)
        self.assertEqual(len(result.ore_positions), 3)
        self.assertTrue(all(int(index.labels[pos]) == base_label for pos in result.ore_positions))
        self.assertEqual(result.diagnostics["base_ore_reachable_fraction"], 1.0)

    def test_raw_mode_can_report_unreachable_task_without_modifying_topology(self):
        mask = np.zeros((8, 8), dtype=bool)
        mask[1, 1] = True
        mask[6, 6] = True
        before = mask.copy()
        config = _config(solvability_mode="raw", count=1, min_base_distance=1)

        result = place_base_and_resources(mask, config, seed=4)

        np.testing.assert_array_equal(mask, before)
        self.assertTrue(result.ok)
        self.assertEqual(result.diagnostics["base_ore_reachable_fraction"], 0.0)

    def test_conditioned_failure_is_structured_when_component_too_small(self):
        mask = np.zeros((10, 10), dtype=bool)
        mask[1, 1] = True
        config = _config(
            solvability_mode="conditioned_same_component",
            min_task_component_fraction=0.5,
        )

        result = place_base_and_resources(mask, config, seed=0)

        self.assertFalse(result.ok)
        self.assertIsNone(result.base_pos)
        self.assertEqual(result.placement_status, "task_placement_failed")
        self.assertEqual(result.diagnostics["placement_failure_reason"], "no_component_meets_threshold")

    def test_minimum_distances_are_enforced(self):
        mask = np.ones((12, 12), dtype=bool)
        config = _config(
            solvability_mode="conditioned_same_component",
            count=3,
            min_base_distance=4,
            min_pair_distance=3,
        )

        result = place_base_and_resources(mask, config, seed=12)

        self.assertTrue(result.ok)
        assert result.base_pos is not None
        for ore_pos in result.ore_positions:
            self.assertGreaterEqual(_manhattan(result.base_pos, ore_pos), 4)
        for left in result.ore_positions:
            for right in result.ore_positions:
                if left != right:
                    self.assertGreaterEqual(_manhattan(left, right), 3)

    def test_clustered_resources_are_reproducible_and_independent_of_uniform(self):
        mask = np.ones((16, 16), dtype=bool)
        clustered = _config(distribution="clustered", count=4, min_pair_distance=2)
        uniform = _config(distribution="uniform", count=4, min_pair_distance=2)

        first = place_base_and_resources(mask, clustered, seed=7)
        second = place_base_and_resources(mask, clustered, seed=7)
        uniform_result = place_base_and_resources(mask, uniform, seed=7)

        self.assertEqual(first, second)
        self.assertNotEqual(first.ore_positions, uniform_result.ore_positions)

    def test_base_margin_is_enforced(self):
        mask = np.ones((12, 12), dtype=bool)
        config = _config(base_margin=3)

        result = place_base_and_resources(mask, config, seed=8)

        self.assertTrue(result.ok)
        assert result.base_pos is not None
        row, col = result.base_pos
        self.assertTrue(3 <= row < 9)
        self.assertTrue(3 <= col < 9)

    def test_rejects_non_2d_mask(self):
        with self.assertRaisesRegex(ValueError, "open_mask"):
            place_base_and_resources(np.array([True, False]), _config(), seed=0)


def _config(
    solvability_mode: str = "conditioned_same_component",
    distribution: str = "uniform",
    count: int = 1,
    min_base_distance: int = 0,
    min_pair_distance: int = 0,
    min_task_component_fraction: float = 0.0,
    base_margin: int = 0,
) -> MapGenerationConfig:
    return MapGenerationConfig.from_config(
        {
            "env": {
                "map_mode": "fractal_percolation",
                "grid_size": [8, 8],
                "world": {
                    "topology": {
                        "model": "iid_site",
                        "p_open": 1.0,
                        "solvability_mode": solvability_mode,
                        "min_task_component_fraction": min_task_component_fraction,
                    },
                    "resources": {
                        "distribution": distribution,
                        "count": count,
                        "hurst": 0.7,
                        "min_base_distance": min_base_distance,
                        "min_pair_distance": min_pair_distance,
                    },
                    "placement": {"base_margin": base_margin, "max_attempts": 100},
                },
            }
        }
    )


def _manhattan(left: tuple[int, int], right: tuple[int, int]) -> int:
    return abs(left[0] - right[0]) + abs(left[1] - right[1])


if __name__ == "__main__":
    unittest.main()
