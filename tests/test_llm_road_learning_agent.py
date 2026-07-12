from __future__ import annotations

import json
import unittest

from evogrid.agents import AgentMemory, LLMRoadLearningAgent
from evogrid.constants import Action, Tile


class LLMRoadLearningAgentTest(unittest.TestCase):
    def test_llm_can_choose_exploratory_build_without_payoff_evidence(self):
        agent = LLMRoadLearningAgent(
            memory=AgentMemory(),
            mock_responses=[_decision(Action.BUILD_ROAD)],
            exploration_budget_per_episode=1,
        )

        action = agent.act(_transport_obs(Tile.ROUGH), {})

        self.assertEqual(action, int(Action.BUILD_ROAD))
        self.assertEqual(agent.trace[-1]["build_decision_source"], "llm_exploration")
        self.assertEqual(agent.trace[-1]["prompt_learned_estimate"]["source"], "none")

    def test_llm_learned_build_uses_prior_episode_records(self):
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
        agent = LLMRoadLearningAgent(
            memory=memory,
            mock_responses=[_decision(Action.BUILD_ROAD)],
            exploration_budget_per_episode=0,
        )

        action = agent.act(_transport_obs(Tile.ROUGH), {})

        self.assertEqual(action, int(Action.BUILD_ROAD))
        self.assertEqual(agent.trace[-1]["build_decision_source"], "llm_learned")
        self.assertGreater(agent.trace[-1]["prompt_learned_estimate"]["learned_value"], 0.0)

    def test_budget_blocks_unlearned_llm_build(self):
        agent = LLMRoadLearningAgent(
            memory=AgentMemory(),
            mock_responses=[_decision(Action.BUILD_ROAD)],
            exploration_budget_per_episode=0,
        )

        action = agent.act(_transport_obs(Tile.ROUGH), {})

        self.assertEqual(action, int(Action.MOVE_LEFT))
        self.assertEqual(agent.trace[-1]["build_decision_source"], "route")
        self.assertFalse(agent.trace[-1]["fallback_used"])
        self.assertEqual(agent.trace[-1]["fallback_reason"], "no_budget_or_learned_signal")
        self.assertEqual(agent.trace[-1]["attempt_count"], 0)

    def test_no_road_learning_hides_prior_records_from_prompt(self):
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
        agent = LLMRoadLearningAgent(
            memory=memory,
            use_road_learning=False,
            mock_responses=[_decision(Action.BUILD_ROAD)],
            exploration_budget_per_episode=1,
        )

        action = agent.act(_transport_obs(Tile.ROUGH), {})

        self.assertEqual(action, int(Action.BUILD_ROAD))
        self.assertEqual(agent.trace[-1]["build_decision_source"], "llm_exploration")
        self.assertEqual(agent.trace[-1]["prompt_learned_estimate"]["source"], "none")

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
        agent = LLMRoadLearningAgent(
            memory=memory,
            mock_responses=[_decision(Action.BUILD_ROAD)],
            exploration_budget_per_episode=0,
            max_learned_builds_per_episode=0,
        )

        action = agent.act(_transport_obs(Tile.ROUGH), {})

        self.assertEqual(action, int(Action.MOVE_LEFT))
        self.assertEqual(agent.trace[-1]["build_decision_source"], "route")
        self.assertEqual(agent.trace[-1]["attempt_count"], 0)

    def test_strict_gate_blocks_weak_contextual_learned_signal(self):
        memory = AgentMemory()
        memory.add_road_credit_records([_context_record(net_payoff=0.5, build_step=0)])
        agent = LLMRoadLearningAgent(
            memory=memory,
            mock_responses=[_decision(Action.BUILD_ROAD)],
            exploration_budget_per_episode=0,
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
        self.assertEqual(agent.trace[-1]["attempt_count"], 0)
        self.assertFalse(agent.trace[-1]["exploration_state"]["learned_evidence_strong"])

    def test_strict_gate_allows_strong_contextual_learned_signal(self):
        memory = AgentMemory()
        memory.add_road_credit_records(
            [_context_record(net_payoff=0.5, build_step=step) for step in range(3)]
        )
        agent = LLMRoadLearningAgent(
            memory=memory,
            mock_responses=[_decision(Action.BUILD_ROAD)],
            exploration_budget_per_episode=0,
            min_contextual_evidence_count=3,
            positive_rate_threshold=0.7,
            learned_value_threshold=0.2,
            confidence_threshold=0.5,
            require_contextual_evidence=True,
            require_on_route_learned_build=True,
        )

        action = agent.act(_transport_obs(Tile.ROUGH), {})

        self.assertEqual(action, int(Action.BUILD_ROAD))
        self.assertEqual(agent.trace[-1]["build_decision_source"], "llm_learned")
        self.assertTrue(agent.trace[-1]["exploration_state"]["learned_evidence_strong"])

    def test_future_use_gate_blocks_late_low_reuse_learned_signal(self):
        memory = AgentMemory()
        memory.add_road_credit_records(
            [_context_record(net_payoff=0.5, build_step=step) for step in range(3)]
        )
        agent = LLMRoadLearningAgent(
            memory=memory,
            mock_responses=[_decision(Action.BUILD_ROAD)],
            exploration_budget_per_episode=0,
            min_contextual_evidence_count=3,
            positive_rate_threshold=0.7,
            learned_value_threshold=0.2,
            confidence_threshold=0.5,
            require_contextual_evidence=True,
            require_on_route_learned_build=True,
            require_future_use_break_even=True,
            future_use_margin=1,
        )

        action = agent.act(_transport_obs(Tile.ROUGH), {"steps_remaining": 0})

        self.assertEqual(action, int(Action.MOVE_LEFT))
        self.assertEqual(agent.trace[-1]["build_decision_source"], "route")
        self.assertEqual(agent.trace[-1]["attempt_count"], 0)
        gate = agent.trace[-1]["exploration_state"]["learned_evidence_gate"]
        self.assertFalse(gate["passes"])
        self.assertIn("estimated_future_uses_below_break_even", gate["failed_reasons"])

    def test_future_use_gate_allows_early_high_reuse_learned_signal(self):
        memory = AgentMemory()
        memory.visited_counts[(1, 1)] = 4
        memory.add_road_credit_records(
            [_context_record(net_payoff=0.5, build_step=step) for step in range(3)]
        )
        agent = LLMRoadLearningAgent(
            memory=memory,
            mock_responses=[_decision(Action.BUILD_ROAD)],
            exploration_budget_per_episode=0,
            min_contextual_evidence_count=3,
            positive_rate_threshold=0.7,
            learned_value_threshold=0.2,
            confidence_threshold=0.5,
            require_contextual_evidence=True,
            require_on_route_learned_build=True,
            require_future_use_break_even=True,
            future_use_margin=1,
        )

        action = agent.act(_transport_obs(Tile.ROUGH), {"steps_remaining": 10})

        self.assertEqual(action, int(Action.BUILD_ROAD))
        self.assertEqual(agent.trace[-1]["build_decision_source"], "llm_learned")
        self.assertEqual(agent.trace[-1]["attempt_count"], 1)
        gate = agent.trace[-1]["exploration_state"]["learned_evidence_gate"]
        self.assertTrue(gate["passes"])


def _decision(action: Action) -> str:
    return json.dumps(
        {
            "mode": "action",
            "action": action.name,
            "action_id": int(action),
            "reason": "test decision",
            "confidence": 1.0,
        }
    )


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
