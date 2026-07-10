from __future__ import annotations

import json
import unittest

from evogrid.agents.deepseek_agent import DeepSeekAgent
from evogrid.agents.hybrid_agent import HybridAgent
from evogrid.constants import Action
from evogrid.envs import EvoGridMineEnv


class DeepSeekAgentTest(unittest.TestCase):
    def test_mock_action_response_maps_to_action_id(self):
        agent = DeepSeekAgent(
            mode="action",
            mock_responses=[json.dumps({"mode": "action", "action": "MOVE_RIGHT", "action_id": 3})],
        )
        env = EvoGridMineEnv()
        obs, info = env.reset(seed=0)
        action = agent.act(obs, info)
        self.assertEqual(action, int(Action.MOVE_RIGHT))

    def test_invalid_response_falls_back(self):
        agent = HybridAgent(mock_responses=["not json"], replan_interval=1)
        env = EvoGridMineEnv()
        obs, info = env.reset(seed=0)
        action = agent.act(obs, info)
        self.assertIsInstance(action, int)
        self.assertTrue(agent.trace[-1]["fallback_used"])
        self.assertIn("llm_error", agent.trace[-1]["fallback_reason"])

    def test_retry_can_recover_from_invalid_response(self):
        agent = DeepSeekAgent(
            mode="action",
            max_retries=1,
            mock_responses=[
                "not json",
                json.dumps({"mode": "action", "action": "MOVE_DOWN", "action_id": 1}),
            ],
        )
        env = EvoGridMineEnv()
        obs, info = env.reset(seed=0)
        action = agent.act(obs, info)
        self.assertEqual(action, int(Action.MOVE_DOWN))
        self.assertFalse(agent.trace[-1]["fallback_used"])
        self.assertEqual(agent.trace[-1]["attempt_count"], 2)
        self.assertTrue(agent.trace[-1]["attempts"][0]["error"])
        self.assertFalse(agent.trace[-1]["attempts"][1]["error"])

    def test_planner_preferred_actions_map_to_action_id(self):
        agent = HybridAgent(
            mock_responses=[
                json.dumps(
                    {
                        "mode": "plan",
                        "subgoal": "move toward ore",
                        "preferred_actions": ["MOVE_RIGHT"],
                    }
                )
            ],
            replan_interval=1,
        )
        env = EvoGridMineEnv()
        obs, info = env.reset(seed=0)
        action = agent.act(obs, info)
        self.assertEqual(action, int(Action.MOVE_RIGHT))


if __name__ == "__main__":
    unittest.main()
