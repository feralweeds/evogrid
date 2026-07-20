from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from scripts.run_skill_evolution_experiment import run_skill_evolution_experiment
from evogrid.visualization.plot_skill_evolution import plot_skill_evolution


class PlotSkillEvolutionTest(unittest.TestCase):
    def test_plots_are_generated_from_saved_outputs(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir) / "m5_smoke"
            run_skill_evolution_experiment("configs/curriculum_self_evolution.yaml", run_dir)
            outputs = plot_skill_evolution(run_dir)

            names = {path.name for path in outputs}
            self.assertEqual(
                names,
                {
                    "verified_skill_count.png",
                    "capability_score.png",
                    "skill_coverage_heatmap.png",
                    "false_trigger_retention.png",
                },
            )
            for path in outputs:
                self.assertTrue(path.exists())
                self.assertGreater(path.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
