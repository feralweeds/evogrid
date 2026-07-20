from __future__ import annotations

import unittest

import numpy as np

from evogrid.envs.chunk_world import ChunkConfig, ChunkWorld, GlobalCoordinateField


class ChunkGlobalFieldTest(unittest.TestCase):
    def test_same_global_points_are_order_independent(self):
        field = GlobalCoordinateField(root_seed=123)
        points = [(-8, 0), (0, 0), (7, 3), (16, -2)]

        first = field.sample_points(points, channel="terrain", hurst=0.5)
        second = field.sample_points(reversed(points), channel="terrain", hurst=0.5)

        self.assertEqual(first, second)

    def test_chunk_generation_order_does_not_change_world_values(self):
        config = ChunkConfig(root_seed=77, chunk_size=8, halo=1, max_cached_chunks=8)
        north_to_south = ChunkWorld(config)
        chunks_a = [north_to_south.get_chunk((0, cy)).roughness.copy() for cy in [-1, 0, 1]]
        south_to_north = ChunkWorld(config)
        chunks_b = [south_to_north.get_chunk((0, cy)).roughness.copy() for cy in [1, 0, -1]]

        self.assertTrue(np.array_equal(chunks_a[0], chunks_b[2]))
        self.assertTrue(np.array_equal(chunks_a[1], chunks_b[1]))
        self.assertTrue(np.array_equal(chunks_a[2], chunks_b[0]))


if __name__ == "__main__":
    unittest.main()
