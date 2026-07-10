from __future__ import annotations

import unittest

from evogrid.constants import Action, Tile
from evogrid.envs import EvoGridMineEnv


class MetricsTest(unittest.TestCase):
    def test_road_usage_rate_is_recorded(self):
        env = EvoGridMineEnv()
        obs, info = env.reset(seed=0)
        env.step(Action.MOVE_RIGHT)
        env.step(Action.BUILD_ROAD)
        env.step(Action.MOVE_RIGHT)
        env.step(Action.MOVE_LEFT)
        obs, reward, terminated, truncated, info = env.step(Action.NOOP)
        self.assertEqual(info["road_cells_built"], 1)
        self.assertGreaterEqual(info["road_usage_rate"], 0.0)

    def test_mining_and_final_carry_state_are_recorded(self):
        env = EvoGridMineEnv(
            {
                "env": {
                    "grid_size": [5, 5],
                    "max_steps": 20,
                    "base_pos": [1, 1],
                    "ore_positions": [[1, 2]],
                    "rough_terrain": [],
                    "obstacles": [],
                }
            }
        )
        env.reset(seed=0)
        obs, reward, terminated, truncated, info = env.step(Action.MINE)
        self.assertEqual(info["num_mine"], 1)
        self.assertTrue(info["final_has_ore"])
        self.assertEqual(info["final_agent_pos"], [1, 1])
        self.assertEqual(info["carrying_steps"], 0)
        self.assertEqual(info["first_mine_step"], 0)

    def test_road_credit_tracks_original_tile_and_payoff(self):
        env = EvoGridMineEnv(
            {
                "env": {
                    "grid_size": [5, 5],
                    "max_steps": 20,
                    "base_pos": [1, 1],
                    "ore_positions": [[3, 3]],
                    "rough_terrain": [[1, 2]],
                    "obstacles": [],
                    "shaping": {"allow_build_road": True, "allow_dig": False},
                }
            }
        )
        env.reset(seed=0)
        env.step(Action.MOVE_RIGHT)
        env.step(Action.BUILD_ROAD)
        env.step(Action.MOVE_LEFT)
        obs, reward, terminated, truncated, info = env.step(Action.MOVE_RIGHT)

        self.assertEqual(info["road_total_usage_count"], 1)
        self.assertAlmostEqual(info["road_saved_cost"], 0.05)
        self.assertAlmostEqual(info["road_build_cost"], 0.1)
        self.assertAlmostEqual(info["road_net_payoff"], -0.05)
        self.assertEqual(len(info["road_credit_records"]), 1)
        record = info["road_credit_records"][0]
        self.assertEqual(record["position"], [1, 2])
        self.assertEqual(record["original_tile"], int(Tile.ROUGH))
        self.assertEqual(record["original_tile_name"], "ROUGH")
        self.assertEqual(record["usage_count"], 1)


if __name__ == "__main__":
    unittest.main()
