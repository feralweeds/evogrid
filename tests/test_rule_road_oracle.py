from __future__ import annotations

import unittest

from evogrid.agents import RuleRoadOracleAgent
from evogrid.constants import Action
from evogrid.envs import EvoGridMineEnv


class RuleRoadOracleTest(unittest.TestCase):
    def test_oracle_builds_road_on_traversed_rough_cell(self):
        env = EvoGridMineEnv(
            {
                "env": {
                    "grid_size": [5, 6],
                    "max_steps": 20,
                    "base_pos": [1, 1],
                    "ore_positions": [[1, 4]],
                    "rough_terrain": [[1, 2], [1, 3]],
                    "obstacles": [],
                    "shaping": {"allow_build_road": True, "allow_dig": False},
                }
            }
        )
        agent = RuleRoadOracleAgent()
        obs, info = env.reset(seed=0)

        first_action = agent.act(obs, info)
        self.assertEqual(first_action, int(Action.MOVE_RIGHT))
        obs, reward, terminated, truncated, info = env.step(first_action)

        second_action = agent.act(obs, info)
        self.assertEqual(second_action, int(Action.BUILD_ROAD))


if __name__ == "__main__":
    unittest.main()
