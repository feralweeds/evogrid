from __future__ import annotations

import unittest

import numpy as np

from evogrid.envs.chunk_world import ChunkConfig, ChunkWorld


class ChunkBoundaryTest(unittest.TestCase):
    def test_east_west_halo_overlap_is_identical(self):
        config = ChunkConfig(root_seed=12, chunk_size=8, halo=1)
        world = ChunkWorld(config)
        left = world.get_chunk((0, 0))
        right = world.get_chunk((1, 0))
        interior_rows = slice(config.halo, config.halo + config.chunk_size)
        left_east_halo = config.halo + config.chunk_size
        right_first_interior = config.halo

        self.assertTrue(
            np.array_equal(
                left.walkable[interior_rows, left_east_halo],
                right.walkable[interior_rows, right_first_interior],
            )
        )
        self.assertTrue(
            np.allclose(
                left.roughness[interior_rows, left_east_halo],
                right.roughness[interior_rows, right_first_interior],
            )
        )
        self.assertTrue(
            np.array_equal(
                left.ore[interior_rows, left_east_halo],
                right.ore[interior_rows, right_first_interior],
            )
        )

    def test_north_south_halo_overlap_is_identical(self):
        config = ChunkConfig(root_seed=12, chunk_size=8, halo=1)
        world = ChunkWorld(config)
        north = world.get_chunk((0, -1))
        south = world.get_chunk((0, 0))
        interior_cols = slice(config.halo, config.halo + config.chunk_size)
        north_south_halo = config.halo + config.chunk_size
        south_first_interior = config.halo

        self.assertTrue(
            np.array_equal(
                north.walkable[north_south_halo, interior_cols],
                south.walkable[south_first_interior, interior_cols],
            )
        )
        self.assertTrue(
            np.allclose(
                north.roughness[north_south_halo, interior_cols],
                south.roughness[south_first_interior, interior_cols],
            )
        )


if __name__ == "__main__":
    unittest.main()
