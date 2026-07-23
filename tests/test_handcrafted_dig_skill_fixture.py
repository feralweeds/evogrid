from __future__ import annotations

import unittest

from evogrid.constants import Tile
from scripts.run_handcrafted_dig_skill_fixture import handcrafted_dig_candidate


class HandcraftedDigSkillFixtureTest(unittest.TestCase):
    def test_candidate_uses_generic_visible_target_selection_and_dig(self):
        spec = handcrafted_dig_candidate()

        self.assertEqual(spec.skill_id, "handcrafted_adjacent_obstacle_dig")
        self.assertEqual(spec.procedure[0]["source"], "visible_tiles")
        self.assertIn(
            {"feature": "candidate.tile_type", "op": "eq", "value": int(Tile.OBSTACLE)},
            spec.procedure[0]["filters"],
        )
        self.assertIn(
            {"feature": "candidate.distance_from_agent", "op": "lte", "value": 1},
            spec.procedure[0]["filters"],
        )
        self.assertEqual(spec.procedure[1]["then"], [{"op": "ACT", "action": "DIG"}])
        self.assertEqual(spec.budget["episode_use_actions"], ["DIG"])


if __name__ == "__main__":
    unittest.main()
