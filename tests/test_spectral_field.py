from __future__ import annotations

import unittest

import numpy as np

from evogrid.envs.map_generation.spectral_field import generate_rank_normalized_field


class SpectralFieldTest(unittest.TestCase):
    def test_shape_and_value_range(self):
        field = generate_rank_normalized_field((16, 24), hurst=0.5, seed=1)

        self.assertEqual(field.shape, (16, 24))
        self.assertEqual(field.dtype, np.float64)
        self.assertGreaterEqual(float(field.min()), 0.0)
        self.assertLessEqual(float(field.max()), 1.0)
        self.assertAlmostEqual(float(field.mean()), 0.5, places=12)

    def test_same_seed_is_identical(self):
        first = generate_rank_normalized_field((32, 32), hurst=0.5, seed=123)
        second = generate_rank_normalized_field((32, 32), hurst=0.5, seed=123)

        np.testing.assert_array_equal(first, second)

    def test_different_seed_changes_field(self):
        first = generate_rank_normalized_field((32, 32), hurst=0.5, seed=123)
        second = generate_rank_normalized_field((32, 32), hurst=0.5, seed=124)

        self.assertFalse(np.array_equal(first, second))

    def test_invalid_hurst_raises(self):
        for hurst in (0.0, 1.0, -0.1, 1.2):
            with self.subTest(hurst=hurst):
                with self.assertRaisesRegex(ValueError, "hurst"):
                    generate_rank_normalized_field((8, 8), hurst=hurst, seed=0)

    def test_invalid_shape_raises(self):
        with self.assertRaisesRegex(ValueError, "shape"):
            generate_rank_normalized_field((0, 8), hurst=0.5, seed=0)

    def test_non_square_maps_are_supported(self):
        field = generate_rank_normalized_field((12, 20), hurst=0.7, seed=5)

        self.assertEqual(field.shape, (12, 20))
        self.assertFalse(np.isnan(field).any())
        self.assertFalse(np.isinf(field).any())

    def test_higher_hurst_has_stronger_neighbor_correlation(self):
        low_values = []
        high_values = []
        for seed in range(8):
            low_values.append(_neighbor_correlation(generate_rank_normalized_field((64, 64), 0.2, seed)))
            high_values.append(_neighbor_correlation(generate_rank_normalized_field((64, 64), 0.8, seed)))

        self.assertGreater(float(np.mean(high_values)), float(np.mean(low_values)) + 0.10)

    def test_no_nan_or_inf(self):
        field = generate_rank_normalized_field((64, 64), hurst=0.8, seed=9)

        self.assertFalse(np.isnan(field).any())
        self.assertFalse(np.isinf(field).any())


def _neighbor_correlation(field: np.ndarray) -> float:
    horizontal_left = field[:, :-1].reshape(-1)
    horizontal_right = field[:, 1:].reshape(-1)
    vertical_top = field[:-1, :].reshape(-1)
    vertical_bottom = field[1:, :].reshape(-1)
    x = np.concatenate([horizontal_left, vertical_top])
    y = np.concatenate([horizontal_right, vertical_bottom])
    return float(np.corrcoef(x, y)[0, 1])


if __name__ == "__main__":
    unittest.main()
