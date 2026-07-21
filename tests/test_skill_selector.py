from __future__ import annotations

import tempfile
import unittest

from evogrid.agents.memory import AgentMemory
from evogrid.agents.random_agent import RandomAgent
from evogrid.agents.route_only_agent import RouteOnlyAgent
from evogrid.agents.skill_agent import SkillAgent
from evogrid.constants import Action, Tile
from evogrid.skills.context import SkillContext
from evogrid.skills.registry import SkillRegistry
from evogrid.skills.runtime import SkillRuntime
from evogrid.skills.schemas import SkillRecord, SkillSpec
from evogrid.skills.selector import SkillSelector
from tests.test_skill_runtime import _call_skill_dict, _child_move_skill_dict, _skill_dict


class SkillSelectorTest(unittest.TestCase):
    def test_selector_defaults_to_verified_only(self):
        selector = SkillSelector()
        verified = SkillRecord(SkillSpec.from_dict(_skill_dict(status="verified")), "verified")
        candidate = SkillRecord(SkillSpec.from_dict(_skill_dict(status="candidate")), "candidate")

        selection = selector.select([candidate, verified], _context())

        self.assertEqual(selection.record.spec.status, "verified")

    def test_candidate_selection_requires_flag(self):
        candidate = SkillRecord(SkillSpec.from_dict(_skill_dict(status="candidate")), "candidate")

        blocked = SkillSelector().select([candidate], _context())
        allowed = SkillSelector(allow_candidates=True).select([candidate], _context())

        self.assertIsNone(blocked.record)
        self.assertEqual(allowed.record.spec.status, "candidate")

    def test_selector_uses_deterministic_priority(self):
        low_data = _skill_dict(status="verified")
        low_data["skill_id"] = "b_skill"
        low_data["budget"]["priority"] = 1
        high_data = _skill_dict(status="verified")
        high_data["skill_id"] = "a_skill"
        high_data["budget"]["priority"] = 2

        selection = SkillSelector().select(
            [
                SkillRecord(SkillSpec.from_dict(low_data), "verified"),
                SkillRecord(SkillSpec.from_dict(high_data), "verified"),
            ],
            _context(),
        )

        self.assertEqual(selection.record.spec.skill_id, "a_skill")

    def test_skill_agent_uses_verified_skill_before_fallback(self):
        with tempfile.TemporaryDirectory() as temp:
            registry = SkillRegistry(temp)
            spec = SkillSpec.from_dict(_skill_dict(status="candidate"))
            record = registry.register_candidate(spec)
            verified = SkillRecord(
                spec=SkillSpec.from_dict({**record.spec.to_dict(), "status": "verified", "spec_hash": record.spec.spec_hash}),
                storage_status="verified",
            )
            registry._write_record("verified", verified)
            runtime = SkillRuntime(
                estimators={
                    "future_route_uses": lambda context, variables: 5,
                    "road_break_even_uses": lambda context, variables: 3,
                }
            )
            agent = SkillAgent(registry, RandomAgent(actions=[int(Action.NOOP)]), runtime=runtime)

            action = agent.act(_obs(), {"steps_remaining": 20, "route_plan": _route_plan()})

            self.assertEqual(action, int(Action.BUILD_ROAD))
            self.assertEqual(agent.trace[0]["source"], "skill")

    def test_skill_agent_keeps_episode_use_limit_across_steps(self):
        with tempfile.TemporaryDirectory() as temp:
            registry = SkillRegistry(temp)
            data = _skill_dict(status="verified")
            data["budget"]["max_uses_per_episode"] = 1
            spec = SkillSpec.from_dict(data)
            registry._write_record("verified", SkillRecord(spec, "verified"))
            runtime = SkillRuntime(
                estimators={
                    "future_route_uses": lambda context, variables: 5,
                    "road_break_even_uses": lambda context, variables: 3,
                }
            )
            agent = SkillAgent(registry, RandomAgent(actions=[int(Action.NOOP)]), runtime=runtime)

            first = agent.act(_obs(), {"steps_remaining": 20, "route_plan": _route_plan()})
            second_obs = {**_obs(), "step": 1}
            second = agent.act(second_obs, {"steps_remaining": 19, "route_plan": _route_plan()})

            self.assertEqual(first, int(Action.BUILD_ROAD))
            self.assertEqual(second, int(Action.NOOP))
            self.assertEqual(agent.trace[-2]["runtime"]["termination"], "episode_use_limit_reached")
            self.assertEqual(agent.trace[-1]["source"], "fallback")

    def test_skill_agent_builds_context_from_route_fallback_hints(self):
        with tempfile.TemporaryDirectory() as temp:
            registry = SkillRegistry(temp)
            data = _skill_dict(status="candidate")
            data["skill_id"] = "fallback_hint_road_building"
            data["applicability"]["all"].append(
                {"feature": "memory.similar_outcome_count", "op": "gte", "value": 3}
            )
            registry.register_candidate(SkillSpec.from_dict(data))
            runtime = SkillRuntime(
                estimators={
                    "future_route_uses": lambda context, variables: 5,
                    "road_break_even_uses": lambda context, variables: 3,
                }
            )
            memory = AgentMemory()
            memory.add_road_credit_records(
                [
                    {
                        "position": [2, 2],
                        "build_step": index,
                        "original_tile": int(Tile.GROUND),
                        "route_on_build": True,
                        "route_remaining_length": 2,
                        "net_payoff": 0.2,
                        "usage_count": 3,
                    }
                    for index in range(3)
                ]
            )
            agent = SkillAgent(
                registry,
                RouteOnlyAgent(memory=memory),
                runtime=runtime,
                allow_candidates=True,
            )

            action = agent.act(_return_to_base_obs(), {"steps_remaining": 20})

            self.assertEqual(action, int(Action.BUILD_ROAD))
            self.assertEqual(agent.trace[0]["source"], "skill")

    def test_skill_agent_falls_back_when_no_skill_applies(self):
        with tempfile.TemporaryDirectory() as temp:
            registry = SkillRegistry(temp)
            agent = SkillAgent(registry, RandomAgent(actions=[int(Action.NOOP)]))

            action = agent.act(_obs(terrain_band="SMOOTH"), {"steps_remaining": 20})

            self.assertEqual(action, int(Action.NOOP))
            self.assertEqual(agent.trace[0]["source"], "fallback")

    def test_skill_agent_resolves_nested_skill_from_registry(self):
        with tempfile.TemporaryDirectory() as temp:
            registry = SkillRegistry(temp)
            child = SkillSpec.from_dict(_child_move_skill_dict(status="verified"))
            parent_data = _call_skill_dict(status="verified", max_depth=1)
            parent_data["budget"]["priority"] = 10
            parent = SkillSpec.from_dict(parent_data)
            registry._write_record("verified", SkillRecord(child, "verified"))
            registry._write_record("verified", SkillRecord(parent, "verified"))
            agent = SkillAgent(registry, RandomAgent(actions=[int(Action.NOOP)]))

            action = agent.act(_obs(), {"steps_remaining": 20, "route_plan": _route_plan()})

            self.assertEqual(action, int(Action.MOVE_RIGHT))
            runtime_trace = agent.trace[0]["runtime"]
            self.assertEqual(runtime_trace["skill_id"], parent.skill_id)
            self.assertEqual(runtime_trace["operations"][0]["child_trace"]["skill_id"], child.skill_id)


def _context() -> SkillContext:
    return SkillContext.from_observable_inputs(
        observation=_obs(),
        info={},
        route_plan={"exists": True, "is_known_transport_route": True, "remaining_length_bucket": "medium"},
    )


def _obs(terrain_band: str = "ROUGH") -> dict:
    return {
        "observation_mode": "partial_obs",
        "agent_pos": [2, 2],
        "base_pos": [1, 1],
        "has_ore": False,
        "step": 0,
        "visible_tiles": [{"pos": [2, 2], "tile": int(Tile.GROUND), "terrain_band": terrain_band}],
    }


def _return_to_base_obs() -> dict:
    return {
        "observation_mode": "partial_obs",
        "agent_pos": [2, 2],
        "base_pos": [2, 3],
        "has_ore": True,
        "step": 0,
        "visible_tiles": [
            {"pos": [2, 2], "tile": int(Tile.GROUND), "terrain_band": "ROUGH"},
            {"pos": [2, 3], "tile": int(Tile.BASE), "terrain_band": "SMOOTH"},
        ],
    }


def _route_plan() -> dict:
    return {"exists": True, "is_known_transport_route": True, "remaining_length_bucket": "medium"}


if __name__ == "__main__":
    unittest.main()
