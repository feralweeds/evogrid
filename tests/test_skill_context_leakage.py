from __future__ import annotations

import unittest

from evogrid.constants import Tile
from evogrid.skills.context import SkillContext
from evogrid.skills.predicates import evaluate_predicate


class SkillContextLeakageTest(unittest.TestCase):
    def test_context_strips_evaluator_fields_recursively(self):
        context = SkillContext.from_observable_inputs(
            observation={
                "agent_pos": [1, 1],
                "base_pos": [1, 1],
                "has_ore": False,
                "visible_tiles": [{"pos": [1, 1], "tile": int(Tile.GROUND), "terrain_band": "NORMAL"}],
                "audit": {"shortest_path_length": 3},
            },
            info={
                "steps_remaining": 5,
                "map_summary": {"ore_positions": [[7, 7]], "agent_pos": [1, 1]},
                "route_rough_tile_count": 8,
            },
            memory_summary={"static_diagnostics": {"largest_component_fraction": 1.0}},
            route_plan={
                "exists": True,
                "hidden": {"ore_positions": [[7, 7]]},
                "observed_tiles": [
                    {
                        "pos": [1, 1],
                        "tile_type": int(Tile.GROUND),
                        "terrain_band": "NORMAL",
                        "audit": {"minimum_cost_path_cost": 1.0},
                    }
                ],
            },
            episode_budget={"steps_remaining": 5},
        )

        self.assertNotIn("audit", context.observation)
        self.assertNotIn("route_rough_tile_count", context.observable_info)
        self.assertNotIn("ore_positions", context.observable_info["map_summary"])
        self.assertNotIn("static_diagnostics", context.memory_summary)
        self.assertNotIn("audit", context.route_plan["observed_tiles"][0])

    def test_hidden_feature_cannot_be_accessed_even_if_nested_input_attempts_it(self):
        context = SkillContext.from_observable_inputs(
            observation={
                "agent_pos": [1, 1],
                "base_pos": [1, 1],
                "has_ore": False,
                "visible_tiles": [{"pos": [1, 1], "tile": int(Tile.GROUND), "terrain_band": "NORMAL"}],
            },
            info={"hidden": {"shortest_path": 4}},
        )

        result = evaluate_predicate(
            {"feature": "hidden.shortest_path", "op": "eq", "value": 4},
            context,
        )

        self.assertFalse(result.ok)
        self.assertIn("feature not allowed", result.errors[0])


if __name__ == "__main__":
    unittest.main()
