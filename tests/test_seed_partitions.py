from __future__ import annotations

import unittest

from evogrid.evaluation.partitions import PARTITION_NAMES, make_seed_partitions


class SeedPartitionTest(unittest.TestCase):
    def test_partitions_are_reproducible_and_expanded_for_manifest(self):
        first = make_seed_partitions(123, {"map": 3, "agent": 2, "bootstrap": 1})
        second = make_seed_partitions(123, {"map": 3, "agent": 2, "bootstrap": 1})

        self.assertEqual(first.to_manifest(), second.to_manifest())
        self.assertEqual(tuple(first.partitions), PARTITION_NAMES)
        self.assertEqual(len(first.partition("train").map_seeds), 3)
        self.assertEqual(len(first.partition("gate").agent_seeds), 2)
        self.assertEqual(len(first.partition("verify").bootstrap_seeds), 1)

    def test_map_agent_bootstrap_namespaces_are_disjoint_by_partition(self):
        partitions = make_seed_partitions(456, {"default": 5})
        partitions.assert_disjoint()
        manifest = partitions.to_manifest()

        self.assertIn("derivation", manifest)
        train_map = set(manifest["partitions"]["train"]["map_seeds"])
        verify_map = set(manifest["partitions"]["verify"]["map_seeds"])
        test_map = set(manifest["partitions"]["test"]["map_seeds"])
        self.assertFalse(train_map & verify_map)
        self.assertFalse(verify_map & test_map)

    def test_negative_size_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "non-negative"):
            make_seed_partitions(1, {"train_map": -1})


if __name__ == "__main__":
    unittest.main()
