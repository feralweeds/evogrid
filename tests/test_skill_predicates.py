from __future__ import annotations

import unittest

from evogrid.constants import Tile
from evogrid.skills.context import SkillContext
from evogrid.skills.predicates import evaluate_predicate


class SkillPredicatesTest(unittest.TestCase):
    def test_all_predicate_matches_observable_context(self):
        context = _context()
        predicate = {
            "all": [
                {"feature": "current.terrain_band", "op": "in", "value": ["ROUGH", "VERY_ROUGH"]},
                {"feature": "route.is_known_transport_route", "op": "eq", "value": True},
                {"feature": "memory.similar_mean_payoff", "op": "gt", "value": 0.0},
            ]
        }

        result = evaluate_predicate(predicate, context)

        self.assertTrue(result.ok)
        self.assertTrue(result.applicable)

    def test_missing_feature_returns_structured_error(self):
        context = _context(terrain_band=None)

        result = evaluate_predicate({"feature": "current.terrain_band", "op": "eq", "value": "ROUGH"}, context)

        self.assertFalse(result.ok)
        self.assertFalse(result.applicable)
        self.assertIn("feature missing", result.errors[0])

    def test_unknown_feature_is_rejected(self):
        result = evaluate_predicate(
            {"feature": "hidden.shortest_path", "op": "eq", "value": 3},
            _context(),
        )

        self.assertFalse(result.ok)
        self.assertIn("feature not allowed", result.errors[0])

    def test_unknown_op_is_rejected(self):
        result = evaluate_predicate(
            {"feature": "cargo.has_ore", "op": "contains_secret", "value": True},
            _context(),
        )

        self.assertFalse(result.ok)
        self.assertIn("op not allowed", result.errors[0])

    def test_not_and_any_nodes(self):
        predicate = {
            "any": [
                {"feature": "cargo.has_ore", "op": "eq", "value": True},
                {"not": {"feature": "route.exists", "op": "eq", "value": False}},
            ]
        }

        self.assertTrue(evaluate_predicate(predicate, _context()).applicable)

    def test_depth_limit_is_enforced(self):
        node = {"feature": "cargo.has_ore", "op": "eq", "value": False}
        for _ in range(10):
            node = {"not": node}

        result = evaluate_predicate(node, _context(), max_depth=3)

        self.assertFalse(result.ok)
        self.assertIn("maximum nesting", result.errors[0])


def _context(terrain_band: str | None = "ROUGH") -> SkillContext:
    visible_item = {"pos": [2, 2], "tile": int(Tile.GROUND)}
    if terrain_band is not None:
        visible_item["terrain_band"] = terrain_band
    return SkillContext.from_observable_inputs(
        observation={
            "observation_mode": "partial_obs",
            "agent_pos": [2, 2],
            "base_pos": [1, 1],
            "has_ore": False,
            "visible_tiles": [visible_item],
        },
        info={"steps_remaining": 40},
        memory_summary={
            "similar_mean_payoff": 0.25,
            "similar_outcome_count": 3,
            "visit_count_bucket": "medium",
        },
        route_plan={
            "exists": True,
            "is_known_transport_route": True,
            "remaining_length_bucket": "medium",
        },
        episode_budget={"steps_remaining": 40},
    )


if __name__ == "__main__":
    unittest.main()
