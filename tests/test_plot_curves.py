from __future__ import annotations

import json
import unittest
from pathlib import Path
import uuid

from evogrid.visualization.plot_curves import plot_first_experiment


class PlotCurvesTest(unittest.TestCase):
    def test_plot_first_experiment_outputs_pngs(self):
        root = Path("outputs") / f"test_plot_curves_{uuid.uuid4().hex}"
        metrics_dir = root / "metrics"
        metrics_dir.mkdir(parents=True)
        csv_path = metrics_dir / "all_eval.csv"
        csv_path.write_text(
            "\n".join(
                [
                    "group,seed,episode,policy_type,model_path,train_log_dir,train_config_path,episode_reward,ore_delivered,num_dig,num_build_road,invalid_actions,road_cells_built,dug_cells,road_usage_rate,transport_steps_per_ore,early_shaping_ratio,late_delivery_rate,steps",
                    "full_shaping,0,0,ppo,,,,1.0,2,1,3,0,3,1,0.5,10,0.1,0.2,100",
                    "no_shaping,0,0,ppo,,,,0.0,1,0,0,2,0,0,0.0,20,0.0,0.1,100",
                ]
            ),
            encoding="utf-8",
        )
        summary_path = root / "summary.json"
        summary_path.write_text(
            json.dumps({"outputs": {"metrics_dir": str(metrics_dir)}}),
            encoding="utf-8",
        )
        outputs = plot_first_experiment(summary_path)
        self.assertEqual(len(outputs), 4)
        for output in outputs:
            self.assertTrue(output.exists())
            self.assertGreater(output.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
