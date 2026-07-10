from __future__ import annotations

import unittest

from evogrid.constants import Action, Tile
from evogrid.envs import EvoGridMineEnv


class EnvStepTest(unittest.TestCase):
    def test_move_and_build_road_changes_map(self):
        env = EvoGridMineEnv()
        obs, info = env.reset(seed=0)
        obs, reward, terminated, truncated, info = env.step(Action.MOVE_RIGHT)
        self.assertFalse(truncated)
        obs, reward, terminated, truncated, info = env.step(Action.BUILD_ROAD)
        row, col = obs["agent_pos"]
        self.assertEqual(Tile(obs["grid"][row][col]), Tile.ROAD)
        self.assertEqual(info["num_build_road"], 1)

    def test_no_shaping_blocks_build_road(self):
        env = EvoGridMineEnv({"env": {"shaping": {"allow_build_road": False, "allow_dig": False}}})
        obs, info = env.reset(seed=0)
        env.step(Action.MOVE_RIGHT)
        obs, reward, terminated, truncated, info = env.step(Action.BUILD_ROAD)
        row, col = obs["agent_pos"]
        self.assertNotEqual(Tile(obs["grid"][row][col]), Tile.ROAD)
        self.assertEqual(info["invalid_actions"], 1)

    def test_dig_adjacent_obstacle(self):
        config = {
            "env": {
                "grid_size": [5, 5],
                "max_steps": 20,
                "base_pos": [1, 1],
                "ore_positions": [[1, 3]],
                "rough_terrain": [],
                "obstacles": [[1, 2]],
                "shaping": {"allow_dig": True, "allow_build_road": True},
            }
        }
        env = EvoGridMineEnv(config)
        obs, info = env.reset(seed=0)
        obs, reward, terminated, truncated, info = env.step(Action.DIG)
        self.assertEqual(Tile(obs["grid"][1][2]), Tile.GROUND)
        self.assertEqual(info["num_dig"], 1)


if __name__ == "__main__":
    unittest.main()

