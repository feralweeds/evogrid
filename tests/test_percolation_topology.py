from __future__ import annotations

import unittest

import numpy as np

from evogrid.envs.map_generation.connectivity import generate_open_mask, label_components
from evogrid.envs.map_generation.schemas import MapGenerationConfig


class PercolationTopologyTest(unittest.TestCase):
    def test_iid_p_open_one_is_all_open(self):
        config = _config(model="iid_site", p_open=1.0)

        mask = generate_open_mask(config, seed=0)

        self.assertEqual(mask.shape, (16, 20))
        self.assertTrue(mask.all())

    def test_iid_mode_keeps_binomial_fluctuation(self):
        config = _config(model="iid_site", p_open=0.5)

        mask = generate_open_mask(config, seed=1)

        self.assertNotEqual(int(mask.sum()), round(0.5 * mask.size))

    def test_correlated_mode_hits_target_open_count(self):
        config = _config(model="correlated_site", p_open=0.65, hurst=0.5)

        mask = generate_open_mask(config, seed=2)

        self.assertEqual(int(mask.sum()), round(0.65 * mask.size))

    def test_same_seed_is_reproducible(self):
        config = _config(model="correlated_site", p_open=0.65, hurst=0.5)

        first = generate_open_mask(config, seed=3)
        second = generate_open_mask(config, seed=3)

        np.testing.assert_array_equal(first, second)

    def test_manual_components_and_spans(self):
        mask = np.array(
            [
                [1, 1, 0, 1],
                [0, 1, 0, 1],
                [1, 0, 0, 1],
                [1, 1, 1, 1],
            ],
            dtype=bool,
        )

        index = label_components(mask)

        self.assertEqual(index.component_count, 2)
        self.assertEqual(sorted(index.component_sizes.values()), [3, 8])
        self.assertIn(index.largest_component_id, index.spans_horizontal)
        self.assertIn(index.largest_component_id, index.spans_vertical)

    def test_raw_mode_does_not_modify_mask_during_labeling(self):
        mask = np.array([[1, 0], [1, 1]], dtype=bool)
        before = mask.copy()

        label_components(mask)

        np.testing.assert_array_equal(mask, before)

    def test_label_components_rejects_non_2d_mask(self):
        with self.assertRaisesRegex(ValueError, "open_mask"):
            label_components(np.array([True, False]))


def _config(model: str, p_open: float, hurst: float | None = None) -> MapGenerationConfig:
    topology = {"model": model, "p_open": p_open}
    if hurst is not None:
        topology["hurst"] = hurst
    return MapGenerationConfig.from_config(
        {
            "env": {
                "map_mode": "fractal_percolation",
                "grid_size": [16, 20],
                "world": {
                    "topology": topology,
                    "resources": {"distribution": "uniform", "count": 1},
                },
            }
        }
    )


if __name__ == "__main__":
    unittest.main()
