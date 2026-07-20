from __future__ import annotations

import csv
import json
import tempfile
from pathlib import Path
import unittest

import yaml

from scripts.calibrate_fractal_maps import run_calibration


class MapCalibrationScriptTest(unittest.TestCase):
    def test_calibration_smoke_writes_manifest_and_diagnostics(self):
        with tempfile.TemporaryDirectory() as temp:
            temp_path = Path(temp)
            config_path = temp_path / "calibration.yaml"
            out_dir = temp_path / "out"
            config_path.write_text(yaml.safe_dump(_small_config(), sort_keys=True), encoding="utf-8")

            manifest = run_calibration(config_path, out_dir, "0:2")

            self.assertEqual(manifest["completion_status"], "completed")
            self.assertEqual(manifest["map_count"], 4)
            self.assertEqual(manifest["planned_cell_count"], 2)
            self.assertEqual(manifest["expected_map_count"], 4)
            self.assertEqual(manifest["integrity"]["duplicate_seed_count"], 0)
            self.assertEqual(manifest["integrity"]["duplicate_cell_seed_count"], 0)
            self.assertIn("runtime", manifest)
            self.assertIn("git", manifest["runtime"])
            self.assertTrue((out_dir / "run_manifest.json").exists())
            self.assertTrue((out_dir / "config_resolved.yaml").exists())
            self.assertTrue((out_dir / "maps" / "map_manifest.jsonl").exists())
            self.assertTrue((out_dir / "maps" / "calibration_gates.json").exists())
            self.assertTrue((out_dir / "maps" / "p_span_curves.csv").exists())
            self.assertTrue((out_dir / "maps" / "p_span_curves.json").exists())
            self.assertTrue((out_dir / "maps" / "p50_summary.csv").exists())
            self.assertTrue((out_dir / "maps" / "p50_summary.json").exists())
            diagnostics_path = out_dir / "maps" / "diagnostics.csv"
            self.assertTrue(diagnostics_path.exists())
            with diagnostics_path.open(encoding="utf-8") as csv_file:
                rows = list(csv.DictReader(csv_file))
            self.assertEqual(len(rows), 4)
            self.assertTrue(all(row["placement_status"] == "ok" for row in rows))
            saved_manifest = json.loads((out_dir / "run_manifest.json").read_text(encoding="utf-8"))
            self.assertFalse(saved_manifest["formal_acceptance"]["passed"])
            self.assertTrue(saved_manifest["reproducibility_check"]["checked"])
            self.assertEqual(saved_manifest["reproducibility_check"]["failure_count"], 0)
            gate_report = json.loads((out_dir / "maps" / "calibration_gates.json").read_text(encoding="utf-8"))
            a1 = next(gate for gate in gate_report["gates"] if gate["gate_id"] == "A1")
            self.assertEqual(a1["details"]["rebuild_hash_checked_count"], 2)
            p_span = json.loads((out_dir / "maps" / "p_span_curves.json").read_text(encoding="utf-8"))
            self.assertEqual(p_span["ci_method"], "wilson")
            self.assertTrue(all("ci_low" in row and "ci_high" in row for row in p_span["curves"]))
            p50 = json.loads((out_dir / "maps" / "p50_summary.json").read_text(encoding="utf-8"))
            self.assertTrue(p50["finite_size_only"])
            self.assertTrue(any(row["p50_interpolation_method"] for row in p50["p50"]))
            for figure in saved_manifest["outputs"]["figures"]:
                self.assertTrue((out_dir / figure).exists())
            self.assertIn("figures/p_span_curves_with_ci.png", saved_manifest["outputs"]["figures"])
            self.assertIn("figures/representative_maps_by_h.png", saved_manifest["outputs"]["figures"])

    def test_duplicate_seed_is_reported_as_integrity_failure(self):
        with tempfile.TemporaryDirectory() as temp:
            temp_path = Path(temp)
            config_path = temp_path / "calibration.yaml"
            out_dir = temp_path / "out"
            config = _small_config()
            config["calibration"]["p_open"] = [0.65]
            config_path.write_text(yaml.safe_dump(config, sort_keys=True), encoding="utf-8")

            manifest = run_calibration(config_path, out_dir, "0,0")

            self.assertEqual(manifest["map_count"], 2)
            self.assertEqual(manifest["integrity"]["duplicate_seed_count"], 1)
            self.assertEqual(manifest["integrity"]["duplicate_cell_seed_count"], 1)
            gate_report = json.loads((out_dir / "maps" / "calibration_gates.json").read_text(encoding="utf-8"))
            a0 = next(gate for gate in gate_report["gates"] if gate["gate_id"] == "A0")
            self.assertFalse(a0["passed"])
            self.assertEqual(a0["details"]["duplicate_cell_seed_count"], 1)


def _small_config() -> dict:
    return {
        "calibration": {
            "schema_version": 1,
            "experiment_type": "map_calibration_smoke",
            "sizes": [[16, 16]],
            "topology_hurst": [0.5],
            "terrain_hurst": [0.7],
            "p_open": [0.65, 0.8],
            "resource_distribution": ["uniform"],
            "reproducibility_sample_count": 2,
        },
        "env": {
            "map_mode": "fractal_percolation",
            "grid_size": [16, 16],
            "world": {
                "schema_version": 1,
                "generator_version": "spectral_fbm_v1",
                "topology": {
                    "model": "correlated_site",
                    "p_open": 0.65,
                    "hurst": 0.5,
                    "solvability_mode": "conditioned_same_component",
                    "task_component": "largest",
                    "min_task_component_fraction": 0.05,
                },
                "terrain": {
                    "hurst": 0.7,
                    "base_move_cost": 0.01,
                    "roughness_strength": 0.04,
                    "cost_exponent": 1.0,
                    "road_move_cost": 0.0,
                    "observation_bins": [0.25, 0.5, 0.75],
                },
                "resources": {
                    "distribution": "uniform",
                    "count": 1,
                    "hurst": 0.7,
                    "min_base_distance": 2,
                    "min_pair_distance": 1,
                },
                "placement": {"base_margin": 1, "max_attempts": 50},
            },
        },
    }


if __name__ == "__main__":
    unittest.main()
