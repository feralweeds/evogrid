from __future__ import annotations

import unittest

from evogrid.constants import Tile
from evogrid.envs.map_builder import build_fixed_map


class MapBuilderTest(unittest.TestCase):
    def test_fixed_map_has_base_and_ore(self):
        grid, base_pos, ore_positions = build_fixed_map()
        self.assertEqual(Tile(grid[base_pos[0]][base_pos[1]]), Tile.BASE)
        self.assertIn((26, 26), ore_positions)
        self.assertEqual(Tile(grid[26][26]), Tile.ORE)

    def test_explicit_obstacles_skip_default_wall(self):
        grid, base_pos, ore_positions = build_fixed_map(
            {
                "env": {
                    "grid_size": [5, 5],
                    "base_pos": [1, 1],
                    "ore_positions": [[1, 3]],
                    "rough_terrain": [],
                    "obstacles": [[1, 2]],
                }
            }
        )
        self.assertEqual(Tile(grid[1][2]), Tile.OBSTACLE)
        self.assertEqual(Tile(grid[1][1]), Tile.BASE)
        self.assertEqual(Tile(grid[1][3]), Tile.ORE)


if __name__ == "__main__":
    unittest.main()

