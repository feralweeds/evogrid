from __future__ import annotations

import unittest

from evogrid.evaluation.fdr import benjamini_hochberg


class FDRTest(unittest.TestCase):
    def test_empty_batch_returns_empty_decisions(self):
        self.assertEqual(benjamini_hochberg({}), [])

    def test_alpha_bounds_are_validated(self):
        with self.assertRaisesRegex(ValueError, "alpha"):
            benjamini_hochberg({"a": 0.1}, alpha=1.0)

    def test_decisions_are_ordered_by_p_value_then_id(self):
        decisions = benjamini_hochberg({"b": 0.01, "a": 0.01, "c": 0.5}, alpha=0.05)

        self.assertEqual([decision.candidate_id for decision in decisions], ["a", "b", "c"])


if __name__ == "__main__":
    unittest.main()
