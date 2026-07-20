from __future__ import annotations

import subprocess
import sys
import textwrap
import unittest

from evogrid.envs.map_generation.seeding import derive_seed


class SeedDerivationTest(unittest.TestCase):
    def test_same_input_is_stable(self):
        self.assertEqual(derive_seed(123, "topology"), derive_seed(123, "topology"))

    def test_label_changes_output(self):
        self.assertNotEqual(derive_seed(0, "topology"), derive_seed(0, "terrain"))

    def test_zero_negative_and_large_root_seeds_are_stable(self):
        self.assertEqual(derive_seed(0), 8103096111329562376)
        self.assertEqual(derive_seed(-7, "agent"), 8256312361378650622)
        self.assertEqual(
            derive_seed(2**80, "placement", {"x": 1}),
            10508772283554159437,
        )

    def test_label_order_is_meaningful(self):
        self.assertEqual(derive_seed(42, "a", "b"), 1234411779224900826)
        self.assertEqual(derive_seed(42, "b", "a"), 10686834073414678690)
        self.assertNotEqual(derive_seed(42, "a", "b"), derive_seed(42, "b", "a"))

    def test_fixed_vectors_for_map_substreams(self):
        self.assertEqual(derive_seed(0, "topology"), 14262569983819786587)
        self.assertEqual(derive_seed(0, "terrain"), 58454895859957871)
        self.assertEqual(derive_seed(123, "topology"), 15905288219945719203)

    def test_rejects_non_json_labels(self):
        with self.assertRaises(TypeError):
            derive_seed(0, object())

    def test_not_affected_by_pythonhashseed(self):
        script = textwrap.dedent(
            """
            from evogrid.envs.map_generation.seeding import derive_seed
            print(derive_seed(123, "topology"))
            """
        )
        outputs = []
        for hash_seed in ("1", "987"):
            completed = subprocess.run(
                [sys.executable, "-c", script],
                check=True,
                capture_output=True,
                env={**__import__("os").environ, "PYTHONHASHSEED": hash_seed},
                text=True,
            )
            outputs.append(completed.stdout.strip())
        self.assertEqual(outputs, ["15905288219945719203", "15905288219945719203"])


if __name__ == "__main__":
    unittest.main()
