import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from late_fusion.common import build_modality_series_for_sample, SampleIndex
from late_fusion.evaluate_predictions import align_predictions, run_evaluation


class TestLateFusionEvaluatePredictions(unittest.TestCase):
    def test_align_predictions_frame_level(self):
        pred = pd.DataFrame(
            {
                "frame_idx": [0, 1, 2, 3],
                "y_pred": [0.0, 1.0, 2.0, 3.0],
                "subject_id": [1, 1, 1, 1],
                "story_id": [1, 1, 1, 1],
            }
        )
        y = np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float64)

        frame_idx, y_true, y_pred, warnings = align_predictions(pred, y)
        self.assertEqual(frame_idx.tolist(), [0, 1, 2, 3])
        self.assertEqual(y_true.tolist(), [0.0, 1.0, 2.0, 3.0])
        self.assertEqual(y_pred.tolist(), [0.0, 1.0, 2.0, 3.0])
        self.assertEqual(warnings, [])

    def test_build_modality_series_for_sample_returns_preprocessed_frame_and_window_series(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            train_ann = root / "train_ann"
            val_ann = root / "val_ann"
            speech_pred = root / "speech_pred"
            transcript_pred = root / "transcript_pred"
            out_dir = root / "out"
            for path in [train_ann, val_ann, speech_pred, transcript_pred, out_dir]:
                path.mkdir(parents=True, exist_ok=True)

            pd.DataFrame({"valence": [0.0, 1.0, 0.0, 1.0]}).to_csv(train_ann / "Subject_1_Story_1.csv", index=False)
            pd.DataFrame({"valence": [0.0, 1.0, 0.0, 1.0]}).to_csv(val_ann / "Subject_1_Story_2.csv", index=False)

            pd.DataFrame(
                {
                    "frame_idx": [0, 1, 2, 3],
                    "timestamp_s": [0.0, 0.04, 0.08, 0.12],
                    "y_pred": [1.0, 2.0, 1.0, 2.0],
                    "subject_id": [1, 1, 1, 1],
                    "story_id": [2, 2, 2, 2],
                    "split": ["val"] * 4,
                    "manifest_id": ["m"] * 4,
                }
            ).to_parquet(speech_pred / "Subject_1_Story_2.parquet", index=False)

            pd.DataFrame(
                {
                    "window_idx": [0, 1],
                    "window_start_frame": [0, 2],
                    "window_end_frame": [1, 3],
                    "window_center_frame": [0, 0],
                    "window_center_s": [0.0, 0.0],
                    "y_pred": [2.0, 2.0],
                    "subject_id": [1, 1],
                    "story_id": [2, 2],
                    "split": ["val", "val"],
                    "manifest_id": ["m", "m"],
                }
            ).to_parquet(transcript_pred / "Subject_1_Story_2.parquet", index=False)

            cfg = {
                "paths": {
                    "train_ann_dir": str(train_ann),
                    "val_ann_dir": str(val_ann),
                    "prediction_dir": str(out_dir),
                },
                "split": {
                    "manifest_id": "late_fusion_test_v1",
                    "subjects_train": [1],
                    "subjects_val": [1],
                    "stories_train": [1],
                    "stories_val": [2],
                },
                "fusion": {
                    "fps": 25.0,
                    "modalities": {
                        "speech": {
                            "prediction_dir": str(speech_pred),
                            "kind": "frame",
                            "weight": 1.0,
                            "cutoff": 0.0,
                            "order": 1,
                        },
                        "transcript": {
                            "prediction_dir": str(transcript_pred),
                            "kind": "window",
                            "weight": 1.0,
                            "cutoff": 0.0,
                            "order": 1,
                        },
                    },
                },
            }

            series = build_modality_series_for_sample(cfg, SampleIndex(subject=1, story=2, split="val"))
            self.assertEqual(set(series.keys()), {"speech", "transcript"})
            self.assertEqual(series["speech"].shape, (4,))
            self.assertEqual(series["transcript"].shape, (4,))
            self.assertAlmostEqual(float(series["speech"].mean()), 0.5, places=5)
            self.assertAlmostEqual(float(series["transcript"].mean()), 0.5, places=5)

    def test_run_evaluation_writes_reports(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            pred_dir = root / "pred"
            ann_dir = root / "ann"
            train_ann_dir = root / "train_ann"
            speech_pred_dir = root / "speech_pred"
            transcript_pred_dir = root / "transcript_pred"
            out_dir = root / "out"
            pred_dir.mkdir(parents=True)
            ann_dir.mkdir(parents=True)
            train_ann_dir.mkdir(parents=True)
            speech_pred_dir.mkdir(parents=True)
            transcript_pred_dir.mkdir(parents=True)

            y_true = np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float32)
            pd.DataFrame({"valence": y_true}).to_csv(ann_dir / "Subject_1_Story_1.csv", index=False)
            pd.DataFrame({"valence": y_true}).to_csv(train_ann_dir / "Subject_1_Story_4.csv", index=False)

            pd.DataFrame(
                {
                    "frame_idx": [0, 1, 2, 3],
                    "timestamp_s": [0.0, 0.04, 0.08, 0.12],
                    "y_pred": [0.0, 1.0, 2.0, 3.0],
                    "subject_id": [1, 1, 1, 1],
                    "story_id": [1, 1, 1, 1],
                    "split": ["val"] * 4,
                    "manifest_id": ["m1"] * 4,
                }
            ).to_parquet(pred_dir / "Subject_1_Story_1.parquet", index=False)
            pd.DataFrame(
                {
                    "frame_idx": [0, 1, 2, 3],
                    "timestamp_s": [0.0, 0.04, 0.08, 0.12],
                    "y_pred": [0.0, 1.0, 2.0, 3.0],
                    "subject_id": [1, 1, 1, 1],
                    "story_id": [1, 1, 1, 1],
                    "split": ["val"] * 4,
                    "manifest_id": ["m1"] * 4,
                }
            ).to_parquet(speech_pred_dir / "Subject_1_Story_1.parquet", index=False)
            pd.DataFrame(
                {
                    "window_idx": [0, 1],
                    "window_start_frame": [0, 2],
                    "window_end_frame": [1, 3],
                    "window_center_frame": [0, 2],
                    "window_center_s": [0.0, 0.08],
                    "y_pred": [0.5, 2.5],
                    "subject_id": [1, 1],
                    "story_id": [1, 1],
                    "split": ["val"] * 2,
                    "manifest_id": ["m1"] * 2,
                }
            ).to_parquet(transcript_pred_dir / "Subject_1_Story_1.parquet", index=False)

            cfg = {
                "paths": {
                    "train_ann_dir": str(train_ann_dir),
                    "prediction_dir": str(pred_dir),
                    "val_ann_dir": str(ann_dir),
                },
                "split": {
                    "manifest_id": "late_fusion_test_v1",
                    "subjects_train": [1],
                    "subjects_val": [1],
                    "stories_train": [4],
                    "stories_val": [1],
                },
                "fusion": {
                    "fps": 25.0,
                    "modalities": {
                        "speech": {
                            "prediction_dir": str(speech_pred_dir),
                            "kind": "frame",
                            "weight": 1.0,
                            "cutoff": 0.0,
                            "order": 1,
                        },
                        "transcript": {
                            "prediction_dir": str(transcript_pred_dir),
                            "kind": "window",
                            "weight": 1.0,
                            "cutoff": 0.0,
                            "order": 1,
                        },
                    },
                },
            }

            metrics_df, summary = run_evaluation(cfg, out_dir, max_plots=1, overwrite=True)
            self.assertEqual(len(metrics_df), 1)
            self.assertGreater(summary["overall_ccc"], 0.999)
            self.assertTrue((out_dir / "metrics_per_sample.csv").exists())
            self.assertTrue((out_dir / "metrics_summary.json").exists())
            self.assertTrue((out_dir / "plots" / "ccc_by_sample.png").exists())
            self.assertTrue((out_dir / "plots" / "distribution_scatter.png").exists())
            self.assertTrue((out_dir / "plots" / "timeseries_subject_1_story_1.png").exists())


if __name__ == "__main__":
    unittest.main()
