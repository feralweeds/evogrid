from __future__ import annotations

import unittest

from evogrid.evaluation.map_calibration_gates import evaluate_map_calibration_gates


class MapCalibrationGatesTest(unittest.TestCase):
    def test_smoke_sized_records_do_not_pass_formal_a0_a1(self):
        report = evaluate_map_calibration_gates(_records(seed_count=2), min_seeds_per_cell=100)

        gates = {gate.gate_id: gate for gate in report.gates}
        self.assertFalse(report.passed)
        self.assertFalse(gates["A0"].passed)
        self.assertFalse(gates["A1"].passed)
        self.assertEqual(gates["A0"].details["min_observed_seeds_per_cell"], 2)

    def test_formal_like_records_pass_basic_gates_when_reproducibility_checked(self):
        report = evaluate_map_calibration_gates(
            _records(seed_count=3),
            min_seeds_per_cell=3,
            reproducibility_check={"checked": True, "checked_count": 3, "failure_count": 0},
        )

        gates = {gate.gate_id: gate for gate in report.gates}
        self.assertTrue(gates["A0"].passed)
        self.assertTrue(gates["A1"].passed)
        self.assertTrue(gates["A2"].passed)
        self.assertTrue(gates["A3"].passed)
        self.assertTrue(gates["A4"].passed)
        self.assertTrue(gates["A5"].passed)
        self.assertEqual(gates["A2"].details["hard_violation_count"], 0)
        self.assertIn("span_probability_groups", gates["A2"].details)
        self.assertIn("cell_reports", gates["A4"].details)
        self.assertEqual(gates["A5"].details["estimable_group_count"], gates["A5"].details["group_count"])
        first_curve = gates["A5"].details["critical_curve_reports"][0]
        self.assertTrue(first_curve["finite_size_only"])
        self.assertAlmostEqual(first_curve["p50"], 0.575)
        first_point = first_curve["points"][0]
        self.assertEqual(first_point["ci_method"], "wilson")
        self.assertLessEqual(first_point["ci_low"], first_point["rate"])
        self.assertGreaterEqual(first_point["ci_high"], first_point["rate"])

    def test_a2_reports_uncertain_nonmonotonic_when_ci_overlaps(self):
        records = _records(seed_count=3)
        for row in records:
            if row["p_open"] == 0.5:
                row["spans_horizontal"] = True
                row["spans_vertical"] = False
            if row["p_open"] == 0.65:
                row["spans_horizontal"] = False
                row["spans_vertical"] = False

        report = evaluate_map_calibration_gates(
            records,
            min_seeds_per_cell=3,
            reproducibility_check={"checked": True, "checked_count": 3, "failure_count": 0},
        )

        a2 = {gate.gate_id: gate for gate in report.gates}["A2"]
        self.assertTrue(a2.details["uncertain_nonmonotonic_count"] > 0)
        self.assertEqual(a2.details["hard_violation_count"], 0)

    def test_a0_fails_when_expected_cell_is_missing(self):
        records = _records(seed_count=3)
        expected_cells = [
            {
                "height": 10,
                "width": 10,
                "topology_hurst": 0.2,
                "terrain_hurst": 0.2,
                "p_open": 0.5,
                "resource_distribution": "uniform",
            },
            {
                "height": 10,
                "width": 10,
                "topology_hurst": 0.9,
                "terrain_hurst": 0.2,
                "p_open": 0.5,
                "resource_distribution": "uniform",
            },
        ]

        report = evaluate_map_calibration_gates(
            records,
            min_seeds_per_cell=3,
            expected_cells=expected_cells,
            reproducibility_check={"checked": True, "checked_count": 3, "failure_count": 0},
        )

        a0 = {gate.gate_id: gate for gate in report.gates}["A0"]
        self.assertFalse(a0.passed)
        self.assertEqual(a0.details["expected_cell_count"], 2)
        self.assertEqual(a0.details["missing_cell_count"], 1)
        self.assertGreater(a0.details["unexpected_cell_count"], 0)

    def test_a1_requires_preregistered_rebuild_sample_count(self):
        report = evaluate_map_calibration_gates(
            _records(seed_count=40),
            min_seeds_per_cell=40,
            reproducibility_check={"checked": True, "checked_count": 5, "failure_count": 0},
        )

        a1 = {gate.gate_id: gate for gate in report.gates}["A1"]
        self.assertFalse(a1.passed)
        self.assertEqual(a1.details["required_rebuild_hash_checked_count"], 30)
        self.assertEqual(a1.details["rebuild_hash_checked_count"], 5)

    def test_a4_fails_when_axis_bias_ci_is_separated(self):
        records = _records(seed_count=30)
        for row in records:
            row["spans_horizontal"] = True
            row["spans_vertical"] = False

        report = evaluate_map_calibration_gates(
            records,
            min_seeds_per_cell=30,
            reproducibility_check={"checked": True, "checked_count": 30, "failure_count": 0},
        )

        a4 = {gate.gate_id: gate for gate in report.gates}["A4"]
        self.assertFalse(a4.passed)
        self.assertGreater(a4.details["significant_axis_bias_count"], 0)

    def test_a4_fails_when_axis_correlation_is_biased(self):
        records = _records(seed_count=30)
        for row in records:
            row["terrain_axis_corr_abs_diff"] = 0.35

        report = evaluate_map_calibration_gates(
            records,
            min_seeds_per_cell=30,
            reproducibility_check={"checked": True, "checked_count": 30, "failure_count": 0},
        )

        a4 = {gate.gate_id: gate for gate in report.gates}["A4"]
        self.assertFalse(a4.passed)
        self.assertFalse(a4.details["axis_correlation_within_tolerance"])
        self.assertGreater(a4.details["max_axis_correlation_abs_diff"], 0.2)

    def test_a3_allows_topology_only_formal_scan(self):
        records = [row for row in _records(seed_count=3) if row["terrain_hurst"] == 0.5]

        report = evaluate_map_calibration_gates(
            records,
            min_seeds_per_cell=3,
            reproducibility_check={"checked": True, "checked_count": 3, "failure_count": 0},
        )

        a3 = {gate.gate_id: gate for gate in report.gates}["A3"]
        self.assertTrue(a3.passed)
        self.assertFalse(a3.details["terrain_axis_applicable"])
        self.assertTrue(a3.details["topology_axis_applicable"])
        self.assertTrue(a3.details["topology_axis_passed"])

    def test_a5_fails_when_p50_is_not_bracketed(self):
        records = _records(seed_count=5)
        for row in records:
            row["spans_horizontal"] = False
            row["spans_vertical"] = False

        report = evaluate_map_calibration_gates(
            records,
            min_seeds_per_cell=5,
            reproducibility_check={"checked": True, "checked_count": 5, "failure_count": 0},
        )

        a5 = {gate.gate_id: gate for gate in report.gates}["A5"]
        self.assertFalse(a5.passed)
        self.assertEqual(a5.details["estimable_group_count"], 0)


def _records(seed_count: int) -> list[dict]:
    rows = []
    for topology_h in [0.2, 0.5, 0.8]:
        for terrain_h in [0.2, 0.5, 0.8]:
            for p_open in [0.5, 0.65, 0.8]:
                for seed in range(seed_count):
                    rows.append(
                        {
                            "schema_version": 1,
                            "height": 10,
                            "width": 10,
                            "topology_hurst": topology_h,
                            "terrain_hurst": terrain_h,
                            "p_open": p_open,
                            "target_p_open": p_open,
                            "realized_p_open": p_open,
                            "resource_distribution": "uniform",
                            "seed": seed,
                            "map_id": f"{topology_h}-{terrain_h}-{p_open}-{seed}",
                            "spans_horizontal": p_open >= 0.65,
                            "spans_vertical": p_open >= 0.65,
                            "estimated_terrain_hurst": terrain_h,
                            "estimated_topology_hurst": topology_h,
                            "terrain_neighbor_correlation": terrain_h,
                            "topology_neighbor_correlation": topology_h,
                            "roughness_mean": 0.5,
                            "terrain_axis_corr_abs_diff": 0.0,
                            "topology_axis_corr_abs_diff": 0.0,
                            "terrain_lag1_semivariance_horizontal": 0.1,
                            "terrain_lag1_semivariance_vertical": 0.1,
                            "topology_lag1_semivariance_horizontal": 0.1,
                            "topology_lag1_semivariance_vertical": 0.1,
                            "placement_status": "ok",
                            "valid_for_percolation_analysis": True,
                        }
                    )
    return rows


if __name__ == "__main__":
    unittest.main()
