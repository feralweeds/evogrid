from __future__ import annotations

import json
import unittest

from evogrid.agents import AgentMemory, ShapingOpportunityBuilder
from evogrid.agents.memory_route_planner import RoutePlan
from evogrid.constants import Action, Tile
from evogrid.llm.prompts import build_self_evolution_messages


class ShapingOpportunityTest(unittest.TestCase):
    def test_rough_tile_outputs_candidate_action_not_action(self):
        memory = AgentMemory()
        obs = _partial_obs(tile=Tile.ROUGH)
        memory.update_from_observation(obs)
        plan = RoutePlan(
            action_id=int(Action.MOVE_RIGHT),
            mode="follow_memory_route",
            next_pos=[2, 3],
            target_pos=[2, 2],
            path=[[2, 2], [2, 3]],
            cost=1.0,
            reason="test route",
        )

        opportunity = ShapingOpportunityBuilder().build(
            obs=obs,
            info={"road_credit_records": []},
            memory=memory,
            route_plan=plan,
            mode="RETURN_BASE",
        )

        self.assertTrue(opportunity["available"])
        self.assertEqual(opportunity["candidate_action"], "BUILD_ROAD")
        self.assertNotIn("action", opportunity)
        self.assertEqual(opportunity["cost"]["break_even_uses"], 2)
        self.assertIn("estimated_future_uses", opportunity["cost"])
        self.assertIn("future_use_surplus", opportunity["cost"])
        self.assertFalse(opportunity["constraints"]["auto_execute"])
        self.assertFalse(opportunity["constraints"]["uses_hidden_map"])
        self.assertFalse(opportunity["constraints"]["uses_future_truth"])

    def test_unroadable_tile_is_unavailable(self):
        opportunity = ShapingOpportunityBuilder().build(
            obs=_partial_obs(tile=Tile.BASE),
            info={},
            memory=AgentMemory(),
            route_plan=None,
            mode="EXPLORE",
        )
        self.assertFalse(opportunity["available"])
        self.assertEqual(opportunity["current_tile"], "BASE")
        self.assertNotIn("candidate_action", opportunity)

    def test_prompt_embeds_candidate_action_as_non_binding_evidence(self):
        opportunity = ShapingOpportunityBuilder().build(
            obs=_partial_obs(tile=Tile.ROUGH),
            info={},
            memory=AgentMemory(),
            route_plan=None,
            mode="EXPLORE",
        )
        messages = build_self_evolution_messages(
            _partial_obs(tile=Tile.ROUGH),
            {},
            AgentMemory().summary(),
            {},
            shaping_opportunity=opportunity,
        )
        payload = json.loads(messages[1]["content"])
        self.assertEqual(payload["shaping_opportunity"]["candidate_action"], "BUILD_ROAD")
        self.assertNotIn("action", payload["shaping_opportunity"])
        self.assertIn("candidate_action is not an instruction", payload["instruction"])


def _partial_obs(tile: Tile) -> dict:
    return {
        "observation_mode": "partial_obs",
        "agent_pos": [2, 2],
        "base_pos": [2, 2],
        "has_ore": False,
        "step": 0,
        "ore_delivered": 0,
        "local_view_radius": 1,
        "local_view_origin": [1, 1],
        "local_view": [
            [int(Tile.GROUND), int(Tile.GROUND), int(Tile.GROUND)],
            [int(Tile.GROUND), int(tile), int(Tile.GROUND)],
            [int(Tile.GROUND), int(Tile.GROUND), int(Tile.GROUND)],
        ],
        "visible_tiles": [
            {"pos": [2, 2], "tile": int(tile)},
            {"pos": [2, 3], "tile": int(Tile.GROUND)},
        ],
        "recent_events": [],
    }


if __name__ == "__main__":
    unittest.main()
