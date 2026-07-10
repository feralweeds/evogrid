from __future__ import annotations

import unittest

from evogrid.agents import AgentMemory, RoadLearningModule, ShapingOpportunityBuilder
from evogrid.agents.memory_route_planner import RoutePlan
from evogrid.agents.road_context import contextualize_road_credit_records
from evogrid.constants import Action, Tile


class RoadLearningTest(unittest.TestCase):
    def test_learning_module_estimates_tile_specific_payoff(self):
        module = RoadLearningModule.from_records(
            [
                {"original_tile": int(Tile.ROUGH), "net_payoff": 1.0, "usage_count": 20},
                {"original_tile": int(Tile.ROUGH), "net_payoff": -0.2, "usage_count": 1},
                {"original_tile": int(Tile.GROUND), "net_payoff": -0.1, "usage_count": 0},
            ]
        )

        rough_estimate = module.estimate(int(Tile.ROUGH))
        ground_estimate = module.estimate(int(Tile.GROUND))

        self.assertEqual(rough_estimate["source"], "tile_specific")
        self.assertEqual(rough_estimate["evidence_count"], 2)
        self.assertAlmostEqual(rough_estimate["learned_value"], 0.4)
        self.assertAlmostEqual(rough_estimate["positive_rate"], 0.5)
        self.assertLess(ground_estimate["learned_value"], 0.0)
        self.assertFalse(rough_estimate["auto_execute"])

    def test_agent_memory_keeps_road_credit_records_across_episodes(self):
        memory = AgentMemory()
        memory.update_from_observation(_obs(step=0))
        memory.update_from_result(
            Action.NOOP,
            0.0,
            _obs(step=1),
            {"road_credit_records": [_record(net_payoff=0.5)]},
            {},
        )
        memory.update_from_observation(_obs(step=0))
        memory.update_from_result(
            Action.NOOP,
            0.0,
            _obs(step=1),
            {"road_credit_records": [_record(net_payoff=0.7)]},
            {},
        )

        self.assertEqual(len(memory.road_credit_records), 2)
        self.assertEqual(memory.road_credit_records[0]["memory_episode"], 0)
        self.assertEqual(memory.road_credit_records[1]["memory_episode"], 1)

    def test_shaping_opportunity_includes_learned_estimate_from_memory(self):
        memory = AgentMemory()
        memory.road_credit_records.append(_record(net_payoff=0.5))
        opportunity = ShapingOpportunityBuilder().build(
            obs=_obs(step=3, tile=Tile.ROUGH),
            info={},
            memory=memory,
        )

        self.assertEqual(opportunity["learned_estimate"]["source"], "tile_specific")
        self.assertGreater(opportunity["learned_estimate"]["learned_value"], 0.0)
        self.assertFalse(opportunity["learned_estimate"]["auto_execute"])

    def test_learning_module_does_not_use_global_prior_as_actionable_estimate(self):
        module = RoadLearningModule.from_records([_record(net_payoff=0.5)])
        ground_estimate = module.estimate(int(Tile.GROUND))

        self.assertEqual(ground_estimate["source"], "none")
        self.assertEqual(ground_estimate["evidence_count"], 0)
        self.assertEqual(ground_estimate["learned_value"], 0.0)
        self.assertGreater(ground_estimate["global_prior"]["sample_count"], 0)

    def test_contextual_estimate_overrides_tile_average(self):
        module = RoadLearningModule.from_records(
            [
                _record(net_payoff=0.6, route_remaining_length=8, route_on_build=True),
                _record(net_payoff=-0.4, route_remaining_length=2, route_on_build=True),
            ]
        )

        medium_route = module.estimate(int(Tile.ROUGH), context=_context(route_remaining_length=7))
        short_route = module.estimate(int(Tile.ROUGH), context=_context(route_remaining_length=2))

        self.assertEqual(medium_route["source"], "contextual")
        self.assertGreater(medium_route["learned_value"], 0.0)
        self.assertEqual(short_route["source"], "contextual")
        self.assertLess(short_route["learned_value"], 0.0)

    def test_contextual_records_do_not_fall_back_to_wrong_context(self):
        module = RoadLearningModule.from_records(
            [_record(net_payoff=0.6, route_remaining_length=8, route_on_build=True)]
        )

        off_route = module.estimate(
            int(Tile.ROUGH),
            context=_context(on_current_route=False, known_as_transport_corridor=False),
        )

        self.assertEqual(off_route["source"], "none")
        self.assertEqual(off_route["evidence_count"], 0)

    def test_shaping_opportunity_uses_contextual_estimate_when_route_context_matches(self):
        memory = AgentMemory()
        memory.road_credit_records.append(
            _record(net_payoff=0.5, route_remaining_length=3, route_on_build=True)
        )
        route_plan = RoutePlan(
            action_id=int(Action.MOVE_RIGHT),
            mode="return_base",
            next_pos=[1, 0],
            target_pos=[1, 0],
            path=[[1, 1], [1, 0]],
            cost=1.0,
            reason="test",
        )
        opportunity = ShapingOpportunityBuilder().build(
            obs=_obs(step=3, tile=Tile.ROUGH),
            info={},
            memory=memory,
            route_plan=route_plan,
            mode="RETURN_BASE",
        )

        self.assertEqual(opportunity["learned_estimate"]["source"], "contextual")
        self.assertGreater(opportunity["learned_estimate"]["learned_value"], 0.0)

    def test_contextualize_records_copies_build_trace_context(self):
        records = [_record(net_payoff=0.5)]
        trace = [
            {
                "step": 0,
                "agent_pos": [1, 1],
                "action": "BUILD_ROAD",
                "build_decision_source": "exploratory",
                "shaping_opportunity": {
                    "route_context": {
                        "on_current_route": True,
                        "mode": "RETURN_BASE",
                        "route_remaining_length": 4,
                    },
                    "memory_evidence": {
                        "known_as_transport_corridor": True,
                        "observed_visit_count": 3,
                    },
                },
            }
        ]

        contextualized = contextualize_road_credit_records(records, trace)

        self.assertTrue(contextualized[0]["route_on_build"])
        self.assertEqual(contextualized[0]["route_remaining_length"], 4)
        self.assertEqual(contextualized[0]["observed_visit_count_on_build"], 3)


def _obs(step: int, tile: Tile = Tile.ROUGH) -> dict:
    return {
        "observation_mode": "partial_obs",
        "agent_pos": [1, 1],
        "base_pos": [1, 1],
        "has_ore": False,
        "step": step,
        "ore_delivered": 0,
        "visible_tiles": [{"pos": [1, 1], "tile": int(tile)}],
    }


def _record(
    net_payoff: float,
    route_remaining_length: int | None = None,
    route_on_build: bool | None = None,
) -> dict:
    record = {
        "position": [1, 1],
        "original_tile": int(Tile.ROUGH),
        "original_tile_name": "ROUGH",
        "build_step": 0,
        "usage_count": 12,
        "net_payoff": net_payoff,
    }
    if route_remaining_length is not None:
        record["route_remaining_length"] = route_remaining_length
    if route_on_build is not None:
        record["route_on_build"] = route_on_build
        record["known_as_transport_corridor"] = route_on_build
    return record


def _context(
    route_remaining_length: int | None = 8,
    on_current_route: bool = True,
    known_as_transport_corridor: bool = True,
) -> dict:
    return {
        "route_context": {
            "on_current_route": on_current_route,
            "route_remaining_length": route_remaining_length,
        },
        "memory_evidence": {
            "known_as_transport_corridor": known_as_transport_corridor,
        },
    }


if __name__ == "__main__":
    unittest.main()
