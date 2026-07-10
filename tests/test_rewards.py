from __future__ import annotations

import unittest

from evogrid.constants import Action
from evogrid.envs import EvoGridMineEnv


class RewardTest(unittest.TestCase):
    def test_invalid_action_reward(self):
        env = EvoGridMineEnv()
        env.reset(seed=0)
        obs, reward, terminated, truncated, info = env.step(Action.BUILD_ROAD)
        self.assertLess(reward, 0)
        self.assertEqual(info["invalid_actions"], 1)

    def test_dropoff_reward(self):
        config = {
            "env": {
                "grid_size": [5, 5],
                "max_steps": 20,
                "base_pos": [1, 1],
                "ore_positions": [[1, 2]],
                "rough_terrain": [],
                "obstacles": [],
            }
        }
        env = EvoGridMineEnv(config)
        env.reset(seed=0)
        env.step(Action.MINE)
        obs, reward, terminated, truncated, info = env.step(Action.DROPOFF)
        self.assertGreater(reward, 9.0)
        self.assertEqual(info["ore_delivered"], 1)


if __name__ == "__main__":
    unittest.main()

