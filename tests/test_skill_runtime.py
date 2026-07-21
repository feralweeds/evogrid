from __future__ import annotations

import unittest

from evogrid.constants import Tile
from evogrid.skills.context import SkillContext
from evogrid.skills.runtime import SkillEpisodeState, SkillRuntime
from evogrid.skills.schemas import SkillSpec


class SkillRuntimeTest(unittest.TestCase):
    def test_handcrafted_fixture_builds_when_break_even_condition_holds(self):
        spec = SkillSpec.from_dict(_skill_dict(status="verified"))
        runtime = SkillRuntime(
            estimators={
                "future_route_uses": lambda context, variables: 5,
                "road_break_even_uses": lambda context, variables: 3,
            }
        )

        result = runtime.execute(spec, _context(), run_id="run", episode_id="ep", step=4)

        self.assertTrue(result.completed)
        self.assertEqual(result.chosen_action, "BUILD_ROAD")
        self.assertEqual(result.trace.to_dict()["chosen_action"], "BUILD_ROAD")
        self.assertTrue(result.trace.observable_context_hash)

    def test_candidate_requires_explicit_allow_flag(self):
        spec = SkillSpec.from_dict(_skill_dict(status="candidate"))

        blocked = SkillRuntime().execute(spec, _context())
        allowed = SkillRuntime(
            estimators={
                "future_route_uses": lambda context, variables: 5,
                "road_break_even_uses": lambda context, variables: 3,
            }
        ).execute(spec, _context(), allow_candidate=True)

        self.assertEqual(blocked.termination, "candidate_not_allowed")
        self.assertEqual(allowed.chosen_action, "BUILD_ROAD")

    def test_not_applicable_predicate_returns_no_action(self):
        spec = SkillSpec.from_dict(_skill_dict(status="verified"))

        result = SkillRuntime().execute(spec, _context(terrain_band="SMOOTH"))

        self.assertEqual(result.termination, "not_applicable")
        self.assertIsNone(result.chosen_action)

    def test_unknown_estimator_safely_terminates(self):
        spec = SkillSpec.from_dict(_skill_dict(status="verified"))

        result = SkillRuntime().execute(spec, _context())

        self.assertEqual(result.termination, "unknown_estimator")

    def test_runtime_budget_is_enforced(self):
        data = _skill_dict(status="verified")
        data["budget"]["max_runtime_steps"] = 1
        spec = SkillSpec.from_dict(data)
        runtime = SkillRuntime(estimators={"future_route_uses": lambda context, variables: 5})

        result = runtime.execute(spec, _context())

        self.assertEqual(result.termination, "runtime_budget_exceeded")

    def test_illegal_action_safely_terminates(self):
        data = _skill_dict(status="verified")
        data["procedure"] = [{"op": "ACT", "action": "BUILD_ROAD"}]
        spec = SkillSpec.from_dict(data)

        result = SkillRuntime().execute(spec, _context(tile=int(Tile.BASE)))

        self.assertEqual(result.termination, "illegal_action")

    def test_episode_use_limit_blocks_second_successful_action(self):
        data = _skill_dict(status="verified")
        data["budget"]["max_uses_per_episode"] = 1
        spec = SkillSpec.from_dict(data)
        episode_state = SkillEpisodeState()
        runtime = SkillRuntime(
            estimators={
                "future_route_uses": lambda context, variables: 5,
                "road_break_even_uses": lambda context, variables: 3,
            }
        )

        first = runtime.execute(spec, _context(), episode_state=episode_state)
        second = runtime.execute(spec, _context(), episode_state=episode_state)

        self.assertEqual(first.chosen_action, "BUILD_ROAD")
        self.assertEqual(second.termination, "episode_use_limit_reached")
        self.assertIsNone(second.chosen_action)

    def test_stop_after_success_blocks_later_invocations(self):
        data = _skill_dict(status="verified")
        data["budget"]["stop_after_success"] = True
        spec = SkillSpec.from_dict(data)
        episode_state = SkillEpisodeState()
        runtime = SkillRuntime(
            estimators={
                "future_route_uses": lambda context, variables: 5,
                "road_break_even_uses": lambda context, variables: 3,
            }
        )

        first = runtime.execute(spec, _context(), episode_state=episode_state)
        second = runtime.execute(spec, _context(), episode_state=episode_state)

        self.assertEqual(first.chosen_action, "BUILD_ROAD")
        self.assertEqual(second.termination, "episode_stop_after_success")
        self.assertIsNone(second.chosen_action)

    def test_unknown_action_is_safe(self):
        data = _skill_dict(status="verified")
        data["procedure"] = [{"op": "ACT", "action": "TELEPORT"}]

        with self.assertRaisesRegex(ValueError, "procedure.ACT.action|Unknown"):
            SkillSpec.from_dict(data)

    def test_hash_mismatch_is_rejected_before_execution(self):
        spec = SkillSpec.from_dict(_skill_dict(status="verified"))
        tampered = SkillSpec(
            schema_version=spec.schema_version,
            skill_id=spec.skill_id,
            version=spec.version,
            status=spec.status,
            name=spec.name,
            description=spec.description,
            problem_addressed=spec.problem_addressed,
            source=spec.source,
            applicability=spec.applicability,
            procedure=[{"op": "RETURN", "result": "tampered"}],
            budget=spec.budget,
            objective=spec.objective,
            dependencies=spec.dependencies,
            rationale=spec.rationale,
            spec_hash=spec.spec_hash,
        )

        with self.assertRaisesRegex(ValueError, "spec_hash"):
            SkillRuntime().execute(tampered, _context())

    def test_select_target_plan_and_follow_visible_route(self):
        spec = SkillSpec.from_dict(_route_skill_dict(status="verified"))

        result = SkillRuntime().execute(spec, _context_with_visible_ore())

        self.assertTrue(result.completed)
        self.assertEqual(result.chosen_action, "MOVE_RIGHT")
        self.assertEqual(result.variables["target"]["pos"], [2, 4])
        self.assertEqual(result.variables["route"]["actions"], ["MOVE_RIGHT", "MOVE_RIGHT"])
        self.assertEqual(result.trace.operations[0]["result"]["tile"], int(Tile.ORE))

    def test_plan_route_requires_unknown_cell_policy(self):
        data = _route_skill_dict(status="verified")
        data["procedure"][1].pop("unknown_cell_policy")
        spec = SkillSpec.from_dict(data)

        result = SkillRuntime().execute(spec, _context_with_visible_ore())

        self.assertEqual(result.termination, "missing_unknown_cell_policy")

    def test_follow_route_requires_max_steps(self):
        data = _route_skill_dict(status="verified")
        data["procedure"][2].pop("max_steps")
        spec = SkillSpec.from_dict(data)

        result = SkillRuntime().execute(spec, _context_with_visible_ore())

        self.assertEqual(result.termination, "missing_follow_route_max_steps")

    def test_call_skill_invokes_resolved_child(self):
        child = SkillSpec.from_dict(_child_move_skill_dict(status="verified"))
        parent = SkillSpec.from_dict(_call_skill_dict(status="verified", max_depth=1))
        runtime = SkillRuntime(skill_resolver=lambda skill_id, version: child if skill_id == child.skill_id else None)

        result = runtime.execute(parent, _context(), run_id="run", episode_id="ep")

        self.assertTrue(result.completed)
        self.assertEqual(result.chosen_action, "MOVE_RIGHT")
        self.assertEqual(result.trace.operations[0]["child_trace"]["skill_id"], child.skill_id)

    def test_call_skill_detects_cycle(self):
        parent = SkillSpec.from_dict(_call_skill_dict(status="verified", max_depth=2))
        runtime = SkillRuntime(skill_resolver=lambda skill_id, version: parent)

        result = runtime.execute(parent, _context())

        self.assertEqual(result.termination, "nested_skill_cycle_detected")

    def test_call_skill_depth_budget_is_enforced(self):
        child = SkillSpec.from_dict(_child_move_skill_dict(status="verified"))
        parent = SkillSpec.from_dict(_call_skill_dict(status="verified", max_depth=0))
        runtime = SkillRuntime(skill_resolver=lambda skill_id, version: child)

        result = runtime.execute(parent, _context())

        self.assertEqual(result.termination, "nested_skill_depth_exceeded")


def _context(terrain_band: str = "ROUGH", tile: int = int(Tile.GROUND)) -> SkillContext:
    return SkillContext.from_observable_inputs(
        observation={
            "agent_pos": [2, 2],
            "base_pos": [1, 1],
            "has_ore": False,
            "visible_tiles": [{"pos": [2, 2], "tile": tile, "terrain_band": terrain_band}],
        },
        info={},
        memory_summary={"similar_mean_payoff": 0.2, "similar_outcome_count": 3, "visit_count_bucket": "medium"},
        route_plan={"exists": True, "is_known_transport_route": True, "remaining_length_bucket": "medium"},
        episode_budget={"steps_remaining": 50},
    )


def _context_with_visible_ore() -> SkillContext:
    return SkillContext.from_observable_inputs(
        observation={
            "agent_pos": [2, 2],
            "has_ore": False,
            "visible_tiles": [
                {"pos": [2, 2], "tile": int(Tile.GROUND), "terrain_band": "SMOOTH"},
                {"pos": [2, 4], "tile": int(Tile.ORE), "terrain_band": "ROUGH"},
                {"pos": [4, 2], "tile": int(Tile.ORE), "terrain_band": "ROUGH"},
            ],
        },
        info={},
        memory_summary={"similar_mean_payoff": 0.2, "similar_outcome_count": 3, "visit_count_bucket": "medium"},
        route_plan={"exists": True, "is_known_transport_route": True, "remaining_length_bucket": "short"},
        episode_budget={"steps_remaining": 20},
    )


def _skill_dict(status: str) -> dict:
    return {
        "schema_version": 1,
        "skill_id": "reusable_road_building",
        "version": "1.0.0",
        "status": status,
        "name": "Reusable road building",
        "description": "Handcrafted fixture for runtime testing.",
        "problem_addressed": "Repeated high-cost transport",
        "source": {"proposer": "fixture", "source_episode_ids": ["fixture/episode/1"]},
        "applicability": {
            "all": [
                {"feature": "current.terrain_band", "op": "in", "value": ["ROUGH", "VERY_ROUGH"]},
                {"feature": "route.is_known_transport_route", "op": "eq", "value": True},
            ]
        },
        "procedure": [
            {"op": "ESTIMATE", "estimator": "future_route_uses", "store_as": "n_use"},
            {"op": "ESTIMATE", "estimator": "road_break_even_uses", "store_as": "n_break_even"},
            {
                "op": "IF",
                "condition": {"left": {"var": "n_use"}, "op": "gte", "right": {"var": "n_break_even"}},
                "then": [{"op": "ACT", "action": "BUILD_ROAD"}],
                "else": [{"op": "RETURN", "result": "not_applicable"}],
            },
        ],
        "budget": {"max_runtime_steps": 4, "max_environment_actions": 1, "max_nested_skill_depth": 0},
        "objective": {"primary_metric": "road_net_payoff", "direction": "maximize"},
        "dependencies": [],
    }


def _route_skill_dict(status: str) -> dict:
    data = _skill_dict(status)
    data.update(
        {
            "skill_id": "visible_ore_route",
            "name": "Visible ore route",
            "applicability": {"all": [{"feature": "route.exists", "op": "eq", "value": True}]},
            "procedure": [
                {
                    "op": "SELECT_TARGET",
                    "source": "visible_tiles",
                    "tile_types": [int(Tile.ORE)],
                    "strategy": "nearest",
                    "store_as": "target",
                },
                {
                    "op": "PLAN_ROUTE",
                    "target_var": "target",
                    "unknown_cell_policy": "avoid",
                    "max_length": 8,
                    "store_as": "route",
                },
                {"op": "FOLLOW_ROUTE", "route_var": "route", "max_steps": 1},
            ],
            "budget": {"max_runtime_steps": 3, "max_environment_actions": 1, "max_nested_skill_depth": 0},
            "objective": {"primary_metric": "ore_route_progress", "direction": "maximize"},
        }
    )
    return data


def _child_move_skill_dict(status: str) -> dict:
    data = _skill_dict(status)
    data.update(
        {
            "skill_id": "child_move_right",
            "name": "Child move right",
            "applicability": {"all": [{"feature": "route.exists", "op": "eq", "value": True}]},
            "procedure": [{"op": "ACT", "action": "MOVE_RIGHT"}],
            "budget": {"max_runtime_steps": 1, "max_environment_actions": 1, "max_nested_skill_depth": 0},
            "objective": {"primary_metric": "route_progress", "direction": "maximize"},
        }
    )
    return data


def _call_skill_dict(status: str, max_depth: int) -> dict:
    data = _skill_dict(status)
    data.update(
        {
            "skill_id": "parent_calls_child",
            "name": "Parent calls child",
            "applicability": {"all": [{"feature": "route.exists", "op": "eq", "value": True}]},
            "procedure": [{"op": "CALL_SKILL", "skill_id": "child_move_right"}],
            "budget": {
                "max_runtime_steps": 1,
                "max_environment_actions": 1,
                "max_nested_skill_depth": max_depth,
            },
            "objective": {"primary_metric": "nested_route_progress", "direction": "maximize"},
        }
    )
    return data


if __name__ == "__main__":
    unittest.main()
