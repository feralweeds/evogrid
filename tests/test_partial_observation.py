from __future__ import annotations

import unittest

from evogrid.agents import AgentMemory, MemoryMapRoutePlanner, PartialGreedyAgent, SelfEvolutionAgent
from evogrid.constants import Action, Tile
from evogrid.envs import EvoGridMineEnv
from evogrid.llm.prompts import build_action_diagnostics, summarize_observation


class PartialObservationTest(unittest.TestCase):
    def test_partial_observation_hides_global_grid_and_ore_positions(self):
        env = EvoGridMineEnv(
            {
                "env": {
                    "grid_size": [8, 8],
                    "base_pos": [1, 1],
                    "ore_positions": [[6, 6]],
                    "rough_terrain": [],
                    "obstacles": [],
                    "observation": {"mode": "partial_obs", "local_view_radius": 2},
                }
            }
        )
        obs, info = env.reset(seed=0)
        self.assertEqual(obs["observation_mode"], "partial_obs")
        self.assertNotIn("grid", obs)
        self.assertNotIn("ore_positions", obs)
        self.assertNotIn("ore_positions", info["map_summary"])
        self.assertEqual(len(obs["local_view"]), 5)
        self.assertEqual(len(obs["local_view"][0]), 5)

    def test_memory_only_learns_visible_ore(self):
        env = EvoGridMineEnv(
            {
                "env": {
                    "grid_size": [8, 8],
                    "base_pos": [1, 1],
                    "ore_positions": [[6, 6]],
                    "rough_terrain": [],
                    "obstacles": [],
                    "observation": {"mode": "partial_obs", "local_view_radius": 2},
                }
            }
        )
        obs, info = env.reset(seed=0)
        memory = AgentMemory()
        memory.update_from_observation(obs)
        self.assertEqual(memory.seen_ore_locations, set())

        visible_env = EvoGridMineEnv(
            {
                "env": {
                    "grid_size": [8, 8],
                    "base_pos": [1, 1],
                    "ore_positions": [[1, 2]],
                    "rough_terrain": [],
                    "obstacles": [],
                    "observation": {"mode": "partial_obs", "local_view_radius": 2},
                }
            }
        )
        obs, info = visible_env.reset(seed=0)
        memory.update_from_observation(obs)
        self.assertIn((1, 2), memory.seen_ore_locations)

    def test_partial_prompt_does_not_include_hidden_ore_positions(self):
        env = EvoGridMineEnv(
            {
                "env": {
                    "grid_size": [8, 8],
                    "base_pos": [1, 1],
                    "ore_positions": [[6, 6]],
                    "rough_terrain": [],
                    "obstacles": [],
                    "observation": {"mode": "partial_obs", "local_view_radius": 2},
                }
            }
        )
        obs, info = env.reset(seed=0)
        summary = summarize_observation(obs)
        self.assertNotIn("ore_positions", summary)
        self.assertEqual(summary["visible_ore_positions"], [])

    def test_action_diagnostics_expose_directional_distance_without_hidden_map(self):
        obs = {
            "observation_mode": "partial_obs",
            "agent_pos": [2, 15],
            "base_pos": [2, 2],
            "has_ore": True,
            "step": 10,
            "ore_delivered": 0,
            "local_view_radius": 1,
            "local_view_origin": [1, 14],
            "local_view": [
                [Tile.GROUND, Tile.GROUND, Tile.GROUND],
                [Tile.GROUND, Tile.GROUND, Tile.GROUND],
                [Tile.GROUND, Tile.GROUND, Tile.GROUND],
            ],
            "visible_tiles": [
                {"pos": [row, col], "tile": int(Tile.GROUND)}
                for row in range(1, 4)
                for col in range(14, 17)
            ],
        }
        diagnostics = {item["action"]: item for item in build_action_diagnostics(obs, {})}
        self.assertEqual(diagnostics["MOVE_LEFT"]["delta_to_base"], -1)
        self.assertEqual(diagnostics["MOVE_UP"]["delta_to_base"], 1)
        self.assertNotIn("ore_positions", diagnostics["MOVE_LEFT"])

    def test_self_evolution_agent_filters_illegal_hidden_mine(self):
        env = EvoGridMineEnv(
            {
                "env": {
                    "grid_size": [8, 8],
                    "base_pos": [1, 1],
                    "ore_positions": [[6, 6]],
                    "rough_terrain": [],
                    "obstacles": [],
                    "observation": {"mode": "partial_obs", "local_view_radius": 2},
                }
            }
        )
        obs, info = env.reset(seed=0)
        agent = SelfEvolutionAgent(
            replan_interval=1,
            mock_responses=['{"mode":"action","action":"MINE","action_id":4,"reason":"try mine"}'],
        )
        action = agent.act(obs, info)
        self.assertNotEqual(action, int(Action.MINE))
        self.assertTrue(agent.trace[-1]["legality_filter_used"])

    def test_memory_records_invalid_result_without_truth(self):
        env = EvoGridMineEnv(
            {
                "env": {
                    "grid_size": [8, 8],
                    "base_pos": [1, 1],
                    "ore_positions": [[6, 6]],
                    "rough_terrain": [],
                    "obstacles": [],
                    "observation": {"mode": "partial_obs", "local_view_radius": 2},
                }
            }
        )
        obs, info = env.reset(seed=0)
        memory = AgentMemory()
        previous_info = dict(info)
        obs, reward, terminated, truncated, info = env.step(Action.MINE)
        memory.update_from_result(Action.MINE, reward, obs, info, previous_info)
        self.assertEqual(len(memory.failed_actions), 1)
        self.assertNotIn((6, 6), memory.seen_ore_locations)

    def test_partial_greedy_does_not_require_global_ore_positions(self):
        env = EvoGridMineEnv(
            {
                "env": {
                    "grid_size": [8, 8],
                    "base_pos": [1, 1],
                    "ore_positions": [[6, 6]],
                    "rough_terrain": [],
                    "obstacles": [],
                    "observation": {"mode": "partial_obs", "local_view_radius": 2},
                }
            }
        )
        obs, info = env.reset(seed=0)
        agent = PartialGreedyAgent()
        action = agent.act(obs, info)
        self.assertIsInstance(action, int)
        self.assertNotEqual(action, int(Action.MINE))
        self.assertEqual(agent.memory.seen_ore_locations, set())

    def test_partial_greedy_mines_visible_adjacent_ore(self):
        env = EvoGridMineEnv(
            {
                "env": {
                    "grid_size": [8, 8],
                    "base_pos": [1, 1],
                    "ore_positions": [[1, 2]],
                    "rough_terrain": [],
                    "obstacles": [],
                    "observation": {"mode": "partial_obs", "local_view_radius": 2},
                }
            }
        )
        obs, info = env.reset(seed=0)
        agent = PartialGreedyAgent()
        self.assertEqual(agent.act(obs, info), int(Action.MINE))
        self.assertIn((1, 2), agent.memory.seen_ore_locations)

    def test_partial_greedy_drops_off_at_base_when_carrying(self):
        obs = {
            "observation_mode": "partial_obs",
            "agent_pos": [1, 1],
            "base_pos": [1, 1],
            "has_ore": True,
            "step": 0,
            "ore_delivered": 0,
            "local_view_radius": 1,
            "local_view_origin": [0, 0],
            "local_view": [
                [Tile.GROUND, Tile.GROUND, Tile.GROUND],
                [Tile.GROUND, Tile.BASE, Tile.GROUND],
                [Tile.GROUND, Tile.GROUND, Tile.GROUND],
            ],
            "visible_tiles": [
                {"pos": [row, col], "tile": int(Tile.BASE if (row, col) == (1, 1) else Tile.GROUND)}
                for row in range(3)
                for col in range(3)
            ],
        }
        agent = PartialGreedyAgent()
        self.assertEqual(agent.act(obs, {}), int(Action.DROPOFF))

    def test_memory_route_planner_digs_adjacent_known_obstacle_when_it_shortens_return(self):
        memory = AgentMemory()
        for col in range(2, 14):
            memory.seen_tiles[(2, col)] = int(Tile.GROUND)
        memory.seen_tiles[(2, 14)] = int(Tile.OBSTACLE)
        memory.seen_tiles[(2, 15)] = int(Tile.GROUND)

        obs = {
            "observation_mode": "partial_obs",
            "agent_pos": [2, 15],
            "base_pos": [2, 2],
            "has_ore": True,
            "step": 0,
            "ore_delivered": 0,
            "visible_tiles": [
                {"pos": [2, 14], "tile": int(Tile.OBSTACLE)},
                {"pos": [2, 15], "tile": int(Tile.GROUND)},
                {"pos": [1, 15], "tile": int(Tile.GROUND)},
                {"pos": [3, 15], "tile": int(Tile.GROUND)},
                {"pos": [2, 16], "tile": int(Tile.GROUND)},
            ],
        }
        plan = MemoryMapRoutePlanner().plan_next_action(obs, memory, target=(2, 2))
        self.assertIsNotNone(plan)
        self.assertEqual(plan.action_id, int(Action.DIG))
        self.assertEqual(plan.next_pos, [2, 14])

    def test_self_evolution_uses_memory_route_planner_when_carrying_ore(self):
        memory = AgentMemory()
        for col in range(2, 14):
            memory.seen_tiles[(2, col)] = int(Tile.GROUND)
        memory.seen_tiles[(2, 14)] = int(Tile.OBSTACLE)
        memory.seen_tiles[(2, 15)] = int(Tile.GROUND)

        obs = {
            "observation_mode": "partial_obs",
            "agent_pos": [2, 15],
            "base_pos": [2, 2],
            "has_ore": True,
            "step": 0,
            "ore_delivered": 0,
            "visible_tiles": [
                {"pos": [2, 14], "tile": int(Tile.OBSTACLE)},
                {"pos": [2, 15], "tile": int(Tile.GROUND)},
                {"pos": [1, 15], "tile": int(Tile.GROUND)},
                {"pos": [3, 15], "tile": int(Tile.GROUND)},
                {"pos": [2, 16], "tile": int(Tile.GROUND)},
            ],
        }
        agent = SelfEvolutionAgent(memory=memory)
        self.assertEqual(agent.act(obs, {}), int(Action.DIG))
        self.assertEqual(agent.route_trace[-1]["planner_mode"], "dig_known_obstacle")


if __name__ == "__main__":
    unittest.main()
