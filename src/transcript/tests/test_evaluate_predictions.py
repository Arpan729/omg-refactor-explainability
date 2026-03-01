import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from transcript.evaluate_predictions import align_transcript_predictions, run_evaluation


class TestTranscriptEvaluatePredictions(unittest.TestCase):
    def test_align_transcript_predictions_window_mean(self):
        pred = pd.DataFrame(
            {
                "window_idx": [0, 1],
                "window_start_frame": [0, 2],
                "window_end_frame": [1, 3],
                "y_pred": [0.5, 2.5],
                "subject_id": [1, 1],
                "story_id": [1, 1],
            }
        )
        y = np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float64)

        window_idx, y_true, y_pred, warnings = align_transcript_predictions(pred, y)
        self.assertEqual(window_idx.tolist(), [0, 1])
        self.assertEqual(y_true.tolist(), [0.5, 2.5])
        self.assertEqual(y_pred.tolist(), [0.5, 2.5])
        self.assertEqual(warnings, [])

    def test_run_evaluation_writes_reports(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            pred_dir = root / "pred"
            ann_dir = root / "ann"
            out_dir = root / "out"
            pred_dir.mkdir(parents=True)
            ann_dir.mkdir(parents=True)

            y_true = np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float32)
            pd.DataFrame({"valence": y_true}).to_csv(ann_dir / "Subject_1_Story_1.csv", index=False)

            pd.DataFrame(
                {
                    "window_idx": [0, 1],
                    "window_start_frame": [0, 2],
                    "window_end_frame": [1, 3],
                    "window_center_frame": [0, 0],
                    "window_center_s": [0.0, 0.0],
                    "y_pred": [0.5, 2.5],
                    "subject_id": [1, 1],
                    "story_id": [1, 1],
                    "split": ["val", "val"],
                    "manifest_id": ["m1", "m1"],
                }
            ).to_parquet(pred_dir / "Subject_1_Story_1.parquet", index=False)

            cfg = {
                "paths": {
                    "prediction_dir": str(pred_dir),
                    "val_ann_dir": str(ann_dir),
                }
            }

            metrics_df, summary = run_evaluation(cfg, out_dir, max_plots=1, overwrite=True)
            self.assertEqual(len(metrics_df), 1)
            self.assertGreater(summary["overall_ccc"], 0.999)
            self.assertTrue((out_dir / "metrics_per_sample.csv").exists())
            self.assertTrue((out_dir / "metrics_summary.json").exists())
            self.assertTrue((out_dir / "plots" / "ccc_by_sample.png").exists())
            self.assertTrue((out_dir / "plots" / "distribution_scatter.png").exists())


if __name__ == "__main__":
    unittest.main()
