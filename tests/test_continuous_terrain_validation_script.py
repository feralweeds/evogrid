from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from evogrid.evaluation.acceptance import audit_run_directory
from scripts.run_continuous_terrain_validation import run_continuous_terrain_validation


class ContinuousTerrainValidationScriptTest(unittest.TestCase):
    def test_validation_writes_e2_gate_report_and_audits(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            out_dir = Path(temp_dir) / "r2_validation"

            manifest = run_continuous_terrain_validation(
                "configs/env_continuous_terrain_formal.yaml",
                out_dir,
                seed=0,
            )
            audit = audit_run_directory(out_dir)

            self.assertTrue(manifest["formal_acceptance"]["passed"])
            self.assertEqual(manifest["formal_acceptance"]["conclusion_level"], "E2")
            self.assertTrue((out_dir / "continuous_terrain_metrics.json").exists())
            self.assertTrue((out_dir / "continuous_terrain_gates.json").exists())
            gate_report = json.loads((out_dir / "continuous_terrain_gates.json").read_text(encoding="utf-8"))
            metrics = json.loads((out_dir / "continuous_terrain_metrics.json").read_text(encoding="utf-8"))
            self.assertTrue(gate_report["passed"])
            self.assertEqual({gate["gate_id"] for gate in gate_report["gates"]}, {"B0", "B1", "B2", "B3"})
            self.assertEqual([row["observed_break_even_uses"] for row in metrics["numeric_cases"]], [10, 4, 2])
            self.assertTrue(audit.passed)
            self.assertEqual(audit.conclusion_level, "E2")
            self.assertTrue(audit.formal_acceptance["validated"])


if __name__ == "__main__":
    unittest.main()
