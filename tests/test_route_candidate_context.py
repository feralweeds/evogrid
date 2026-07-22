from __future__ import annotations

import unittest

from evogrid.agents.memory import AgentMemory
from evogrid.agents.memory_route_planner import RoutePlan
from evogrid.agents.route_only_agent import RouteOnlyAgent
from evogrid.constants import Action, Tile


class RouteCandidateContextTest(unittest.TestCase):
    def test_skill_context_hints_include_only_observed_route_tiles(self):
        agent = RouteOnlyAgent(memory=AgentMemory(), route_planner=_StubPlanner())

        hints = agent.skill_context_hints(_obs(), {"steps_remaining": 20})

        observed = hints["route_plan"]["observed_tiles"]
        self.assertEqual([item["pos"] for item in observed], [[2, 2], [2, 3], [2, 4]])
        self.assertEqual([item["route_order"] for item in observed], [0, 1, 2])
        self.assertEqual([item["distance_from_agent"] for item in observed], [0, 1, 2])
        self.assertEqual(observed[0]["terrain_band"], "ROUGH")
        self.assertEqual(observed[0]["tile_type"], int(Tile.GROUND))
        self.assertFalse(observed[0]["has_road"])
        self.assertTrue(observed[2]["has_road"])
        self.assertNotIn([9, 9], [item["pos"] for item in observed])

    def test_missing_route_returns_empty_observed_tiles(self):
        agent = RouteOnlyAgent(memory=AgentMemory(), route_planner=_NoRoutePlanner())

        hints = agent.skill_context_hints(_obs_without_ore(), {"steps_remaining": 20})

        self.assertFalse(hints["route_plan"]["exists"])
        self.assertEqual(hints["route_plan"]["observed_tiles"], [])


class _StubPlanner:
    def plan_next_action(self, obs, memory, target, allow_dig=True, allow_unknown=True):
        return RoutePlan(
            action_id=int(Action.MOVE_RIGHT),
            mode="follow_memory_route",
            next_pos=[2, 3],
            target_pos=list(target),
            path=[[2, 2], [2, 3], [2, 4], [9, 9]],
            cost=3.0,
            reason="test route",
        )


class _NoRoutePlanner:
    def plan_next_action(self, obs, memory, target, allow_dig=True, allow_unknown=True):
        return None


def _obs() -> dict:
    return {
        "observation_mode": "partial_obs",
        "agent_pos": [2, 2],
        "base_pos": [1, 1],
        "has_ore": False,
        "step": 0,
        "visible_tiles": [
            {"pos": [2, 2], "tile": int(Tile.GROUND), "terrain_band": "ROUGH"},
            {"pos": [2, 3], "tile": int(Tile.ROUGH), "terrain_band": "VERY_ROUGH"},
            {"pos": [2, 4], "tile": int(Tile.ROAD), "terrain_band": "NORMAL"},
            {"pos": [2, 5], "tile": int(Tile.ORE), "terrain_band": "NORMAL"},
        ],
    }


def _obs_without_ore() -> dict:
    data = _obs()
    data["visible_tiles"] = data["visible_tiles"][:3]
    return data


if __name__ == "__main__":
    unittest.main()
