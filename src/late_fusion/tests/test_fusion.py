import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from late_fusion.common import (
    SampleIndex,
    align_series_to_length,
    f_trick,
    fuse_sample_predictions,
    reconstruct_frame_series_from_windows,
)


class TestFusion(unittest.TestCase):
    def test_f_trick_matches_target_mean_and_std(self):
        y_train = np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float32)
        preds = np.array([10.0, 20.0, 30.0, 40.0], dtype=np.float32)

        adjusted = f_trick(y_train, preds)

        self.assertAlmostEqual(float(adjusted.mean()), float(y_train.mean()), places=6)
        self.assertAlmostEqual(float(adjusted.std()), float(y_train.std()), places=6)

    def test_reconstruct_frame_series_from_windows_averages_overlaps(self):
        pred_df = pd.DataFrame(
            {
                "window_idx": [0, 1],
                "window_start_frame": [0, 2],
                "window_end_frame": [3, 5],
                "y_pred": [1.0, 3.0],
                "subject_id": [1, 1],
                "story_id": [1, 1],
            }
        )

        series = reconstruct_frame_series_from_windows(pred_df, target_len=6)

        np.testing.assert_allclose(series, np.array([1.0, 1.0, 2.0, 2.0, 3.0, 3.0], dtype=np.float32))

    def test_align_series_to_length_uses_interpolation(self):
        series = np.array([0.0, 10.0], dtype=np.float32)
        aligned = align_series_to_length(series, target_len=4)
        self.assertEqual(aligned.shape, (4,))
        self.assertTrue(np.all(np.isfinite(aligned)))

    def test_fuse_sample_predictions_applies_weights_and_tricks(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            train_ann = root / "train_ann"
            val_ann = root / "val_ann"
            mod1 = root / "mod1"
            mod2 = root / "mod2"
            out_dir = root / "out"
            for path in [train_ann, val_ann, mod1, mod2, out_dir]:
                path.mkdir(parents=True, exist_ok=True)

            pd.DataFrame({"valence": [0.0, 1.0, 0.0, 1.0]}).to_csv(train_ann / "Subject_1_Story_1.csv", index=False)
            pd.DataFrame({"valence": [0.0, 1.0, 0.0, 1.0]}).to_csv(val_ann / "Subject_1_Story_2.csv", index=False)

            frame_pred = pd.DataFrame(
                {
                    "frame_idx": [0, 1, 2, 3],
                    "timestamp_s": [0.0, 0.04, 0.08, 0.12],
                    "y_pred": [1.0, 2.0, 1.0, 2.0],
                    "subject_id": [1, 1, 1, 1],
                    "story_id": [2, 2, 2, 2],
                    "split": ["val"] * 4,
                    "manifest_id": ["m"] * 4,
                }
            )
            frame_pred.to_parquet(mod1 / "Subject_1_Story_2.parquet", index=False)

            window_pred = pd.DataFrame(
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
            )
            window_pred.to_parquet(mod2 / "Subject_1_Story_2.parquet", index=False)

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
                    "final_cutoff": 0.0,
                    "final_order": 1,
                    "modalities": {
                        "mod1": {
                            "prediction_dir": str(mod1),
                            "kind": "frame",
                            "weight": 1.0,
                            "cutoff": 0.0,
                            "order": 1,
                        },
                        "mod2": {
                            "prediction_dir": str(mod2),
                            "kind": "window",
                            "weight": 1.0,
                            "cutoff": 0.0,
                            "order": 1,
                        },
                    },
                },
                "predict": {"device": "cpu"},
            }

            fused, transformed = fuse_sample_predictions(cfg, SampleIndex(subject=1, story=2, split="val"))

            self.assertEqual(set(transformed.keys()), {"mod1", "mod2"})
            self.assertEqual(fused.shape, (4,))
            self.assertTrue(np.all(np.isfinite(fused)))
            self.assertAlmostEqual(float(fused.mean()), 0.5, places=5)


if __name__ == "__main__":
    unittest.main()
