from __future__ import annotations

import unittest

from evogrid.agents import AgentMemory, ExplorationRoadAgent, LearnedRoadAgent
from evogrid.constants import Action, Tile


class LearnedRoadAgentTest(unittest.TestCase):
    def test_cold_start_does_not_build_without_payoff_evidence(self):
        agent = LearnedRoadAgent(memory=AgentMemory())
        obs = _obs(Tile.ROUGH)
        action = agent.act(obs, {})
        self.assertNotEqual(action, int(Action.BUILD_ROAD))
        self.assertEqual(agent.trace[-1]["learned_evidence_count"], 0)

    def test_positive_learned_value_builds_candidate_road(self):
        memory = AgentMemory()
        memory.add_road_credit_records(
            [
                {
                    "position": [1, 1],
                    "original_tile": int(Tile.ROUGH),
                    "build_step": 0,
                    "usage_count": 10,
                    "net_payoff": 0.5,
                }
            ]
        )
        agent = LearnedRoadAgent(memory=memory)
        action = agent.act(_obs(Tile.ROUGH), {})
        self.assertEqual(action, int(Action.BUILD_ROAD))
        self.assertGreater(agent.trace[-1]["learned_value"], 0.0)

    def test_other_tile_evidence_does_not_trigger_build(self):
        memory = AgentMemory()
        memory.add_road_credit_records(
            [
                {
                    "position": [1, 1],
                    "original_tile": int(Tile.ROUGH),
                    "build_step": 0,
                    "usage_count": 10,
                    "net_payoff": 0.5,
                }
            ]
        )
        agent = LearnedRoadAgent(memory=memory)
        action = agent.act(_obs(Tile.GROUND), {})
        self.assertNotEqual(action, int(Action.BUILD_ROAD))
        self.assertEqual(agent.trace[-1]["learned_evidence_count"], 0)


class ExplorationRoadAgentTest(unittest.TestCase):
    def test_epsilon_exploration_can_build_without_payoff_evidence(self):
        agent = ExplorationRoadAgent(
            memory=AgentMemory(),
            epsilon=1.0,
            uncertainty_epsilon=0.0,
        )
        action = agent.act(_transport_obs(Tile.ROUGH), {})

        self.assertEqual(action, int(Action.BUILD_ROAD))
        self.assertEqual(agent.trace[-1]["build_decision_source"], "exploratory")
        self.assertEqual(agent.trace[-1]["learned_evidence_count"], 0)

    def test_zero_exploration_probability_follows_route(self):
        agent = ExplorationRoadAgent(
            memory=AgentMemory(),
            epsilon=0.0,
            uncertainty_epsilon=0.0,
        )
        action = agent.act(_transport_obs(Tile.ROUGH), {})

        self.assertEqual(action, int(Action.MOVE_LEFT))
        self.assertEqual(agent.trace[-1]["build_decision_source"], "route")

    def test_positive_estimate_uses_learned_source_before_exploration(self):
        memory = AgentMemory()
        memory.add_road_credit_records(
            [
                {
                    "position": [1, 1],
                    "original_tile": int(Tile.ROUGH),
                    "build_step": 0,
                    "usage_count": 10,
                    "net_payoff": 0.5,
                }
            ]
        )
        agent = ExplorationRoadAgent(memory=memory, epsilon=1.0)
        action = agent.act(_transport_obs(Tile.ROUGH), {})

        self.assertEqual(action, int(Action.BUILD_ROAD))
        self.assertEqual(agent.trace[-1]["build_decision_source"], "learned")

    def test_strict_gate_blocks_single_contextual_success(self):
        memory = AgentMemory()
        memory.add_road_credit_records([_context_record(net_payoff=0.5, build_step=0)])
        agent = ExplorationRoadAgent(
            memory=memory,
            epsilon=0.0,
            uncertainty_epsilon=0.0,
            min_contextual_evidence_count=3,
            positive_rate_threshold=0.7,
            learned_value_threshold=0.2,
            confidence_threshold=0.5,
            require_contextual_evidence=True,
            require_on_route_learned_build=True,
        )
        action = agent.act(_transport_obs(Tile.ROUGH), {})

        self.assertEqual(action, int(Action.MOVE_LEFT))
        self.assertEqual(agent.trace[-1]["build_decision_source"], "route")
        self.assertFalse(agent.trace[-1]["learned_evidence_strong"])

    def test_strict_gate_allows_repeated_contextual_success(self):
        memory = AgentMemory()
        memory.add_road_credit_records(
            [_context_record(net_payoff=0.5, build_step=step) for step in range(3)]
        )
        agent = ExplorationRoadAgent(
            memory=memory,
            epsilon=0.0,
            uncertainty_epsilon=0.0,
            min_contextual_evidence_count=3,
            positive_rate_threshold=0.7,
            learned_value_threshold=0.2,
            confidence_threshold=0.5,
            require_contextual_evidence=True,
            require_on_route_learned_build=True,
        )
        action = agent.act(_transport_obs(Tile.ROUGH), {})

        self.assertEqual(action, int(Action.BUILD_ROAD))
        self.assertEqual(agent.trace[-1]["build_decision_source"], "learned")
        self.assertTrue(agent.trace[-1]["learned_evidence_strong"])

    def test_learned_build_cap_blocks_positive_estimate(self):
        memory = AgentMemory()
        memory.add_road_credit_records(
            [
                {
                    "position": [1, 1],
                    "original_tile": int(Tile.ROUGH),
                    "build_step": 0,
                    "usage_count": 10,
                    "net_payoff": 0.5,
                }
            ]
        )
        agent = ExplorationRoadAgent(
            memory=memory,
            epsilon=0.0,
            uncertainty_epsilon=0.0,
            max_learned_builds_per_episode=0,
        )
        action = agent.act(_transport_obs(Tile.ROUGH), {})

        self.assertEqual(action, int(Action.MOVE_LEFT))
        self.assertEqual(agent.trace[-1]["build_decision_source"], "route")


def _obs(tile: Tile) -> dict:
    return {
        "observation_mode": "partial_obs",
        "agent_pos": [1, 1],
        "base_pos": [1, 1],
        "has_ore": False,
        "step": 0,
        "ore_delivered": 0,
        "visible_tiles": [
            {"pos": [1, 1], "tile": int(tile)},
            {"pos": [1, 2], "tile": int(Tile.GROUND)},
        ],
    }


def _transport_obs(tile: Tile) -> dict:
    return {
        "observation_mode": "partial_obs",
        "agent_pos": [1, 1],
        "base_pos": [1, 0],
        "has_ore": True,
        "step": 0,
        "ore_delivered": 0,
        "visible_tiles": [
            {"pos": [1, 1], "tile": int(tile)},
            {"pos": [1, 0], "tile": int(Tile.BASE)},
            {"pos": [1, 2], "tile": int(Tile.GROUND)},
        ],
    }


def _context_record(net_payoff: float, build_step: int) -> dict:
    return {
        "position": [1, 1],
        "original_tile": int(Tile.ROUGH),
        "build_step": build_step,
        "usage_count": 10,
        "net_payoff": net_payoff,
        "route_on_build": True,
        "known_as_transport_corridor": True,
        "route_remaining_length": 2,
    }


if __name__ == "__main__":
    unittest.main()
