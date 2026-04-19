import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from ensemble_fusion.common import (
    SampleIndex,
    build_feature_frame,
    butter_lowpass_filter_bidirectional,
    ccc_numpy,
    predict_sample,
    train_model,
    write_prediction_parquet,
)


class TestModel(unittest.TestCase):
    def test_bidirectional_smoothing_preserves_shape(self):
        series = np.array([0.0, 1.0, 0.0, 1.0, 0.0], dtype=np.float32)
        smoothed = butter_lowpass_filter_bidirectional(series, cutoff=0.5, fs=25.0, order=1)
        self.assertEqual(smoothed.shape, series.shape)
        self.assertTrue(np.all(np.isfinite(smoothed)))

    def test_train_and_predict_end_to_end(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            train_ann = root / "train_ann"
            val_ann = root / "val_ann"
            ckpt_dir = root / "checkpoints"
            pred_out_raw = root / "predictions_raw"
            pred_out_smoothed = root / "predictions_smoothed"
            for path in [train_ann, val_ann, ckpt_dir, pred_out_raw, pred_out_smoothed]:
                path.mkdir(parents=True, exist_ok=True)

            mod_names = ["speech", "transcript", "landmarks", "raw_face", "fullbody"]
            pred_dirs = {}
            oof_dirs = {}
            for name in mod_names:
                pred_dirs[name] = root / f"{name}_pred"
                oof_dirs[name] = root / f"{name}_oof"
                pred_dirs[name].mkdir(parents=True, exist_ok=True)
                oof_dirs[name].mkdir(parents=True, exist_ok=True)

            y_train = np.array([0.0, 0.2, 0.4, 0.6], dtype=np.float32)
            y_val = np.array([0.1, 0.3, 0.5, 0.7], dtype=np.float32)
            pd.DataFrame({"valence": y_train}).to_csv(train_ann / "Subject_1_Story_1.csv", index=False)
            pd.DataFrame({"valence": y_val}).to_csv(val_ann / "Subject_1_Story_2.csv", index=False)

            frame_base = {
                "frame_idx": [0, 1, 2, 3],
                "timestamp_s": [0.0, 0.04, 0.08, 0.12],
                "subject_id": [1, 1, 1, 1],
                "story_id": [1, 1, 1, 1],
                "split": ["train"] * 4,
                "manifest_id": ["m"] * 4,
            }
            val_frame_base = {
                "frame_idx": [0, 1, 2, 3],
                "timestamp_s": [0.0, 0.04, 0.08, 0.12],
                "subject_id": [1, 1, 1, 1],
                "story_id": [2, 2, 2, 2],
                "split": ["val"] * 4,
                "manifest_id": ["m"] * 4,
            }
            transcript_train = pd.DataFrame(
                {
                    "window_idx": [0, 1],
                    "window_start_frame": [0, 2],
                    "window_end_frame": [1, 3],
                    "y_pred": [0.1, 0.5],
                    "subject_id": [1, 1],
                    "story_id": [1, 1],
                    "split": ["train", "train"],
                    "manifest_id": ["m", "m"],
                }
            )
            transcript_val = pd.DataFrame(
                {
                    "window_idx": [0, 1],
                    "window_start_frame": [0, 2],
                    "window_end_frame": [1, 3],
                    "y_pred": [0.2, 0.6],
                    "subject_id": [1, 1],
                    "story_id": [2, 2],
                    "split": ["val", "val"],
                    "manifest_id": ["m", "m"],
                }
            )

            for idx, name in enumerate(mod_names):
                if name == "transcript":
                    transcript_train.to_parquet(oof_dirs[name] / "Subject_1_Story_1.parquet", index=False)
                    transcript_val.to_parquet(pred_dirs[name] / "Subject_1_Story_2.parquet", index=False)
                    continue

                train_df = pd.DataFrame({**frame_base, "y_pred": (y_train + 0.01 * idx).tolist()})
                val_df = pd.DataFrame({**val_frame_base, "y_pred": (y_val + 0.01 * idx).tolist()})
                train_df.to_parquet(oof_dirs[name] / "Subject_1_Story_1.parquet", index=False)
                val_df.to_parquet(pred_dirs[name] / "Subject_1_Story_2.parquet", index=False)

            cfg = {
                "paths": {
                    "train_ann_dir": str(train_ann),
                    "val_ann_dir": str(val_ann),
                    "checkpoint_dir": str(ckpt_dir),
                    "prediction_dir_raw": str(pred_out_raw),
                    "prediction_dir_smoothed": str(pred_out_smoothed),
                },
                "split": {
                    "manifest_id": "ensemble_fusion_test_v1",
                    "subjects_train": [1],
                    "subjects_val": [1],
                    "stories_train": [1],
                    "stories_val": [2],
                },
                "fusion": {
                    "fps": 25.0,
                    "final_cutoff": 0.5,
                    "final_order": 1,
                    "modalities": {
                        "speech": {"prediction_dir": str(pred_dirs["speech"]), "oof_dir": str(oof_dirs["speech"]), "kind": "frame", "cutoff": 0.5, "order": 1},
                        "transcript": {"prediction_dir": str(pred_dirs["transcript"]), "oof_dir": str(oof_dirs["transcript"]), "kind": "window", "cutoff": 0.5, "order": 1},
                        "landmarks": {"prediction_dir": str(pred_dirs["landmarks"]), "oof_dir": str(oof_dirs["landmarks"]), "kind": "frame", "cutoff": 0.5, "order": 1},
                        "raw_face": {"prediction_dir": str(pred_dirs["raw_face"]), "oof_dir": str(oof_dirs["raw_face"]), "kind": "frame", "cutoff": 0.5, "order": 1},
                        "fullbody": {"prediction_dir": str(pred_dirs["fullbody"]), "oof_dir": str(oof_dirs["fullbody"]), "kind": "frame", "cutoff": 0.5, "order": 1},
                    },
                },
                "model": {"positive": True, "fit_intercept": True, "max_iter": 10000},
                "train": {
                    "alpha": 0.0001,
                    "l1_ratio": 0.5,
                    "seed": 42,
                    "checkpoint_name": "model.pkl",
                },
                "predict": {"device": "cpu"},
            }

            x_sample, _, names = build_feature_frame(cfg, SampleIndex(subject=1, story=1, split="train"), source="oof")
            self.assertEqual(x_sample.shape, (4, 5))
            self.assertEqual(len(names), 5)

            ckpt_path, metrics = train_model(cfg)
            self.assertTrue(ckpt_path.exists())
            self.assertGreater(metrics["train_ccc"], 0.8)

            y_pred_raw, y_pred_smoothed, _ = predict_sample(cfg, SampleIndex(subject=1, story=2, split="val"), ckpt_path)
            self.assertEqual(y_pred_raw.shape, (4,))
            self.assertEqual(y_pred_smoothed.shape, (4,))
            self.assertTrue(np.all(np.isfinite(y_pred_raw)))
            self.assertTrue(np.all(np.isfinite(y_pred_smoothed)))
            self.assertGreater(ccc_numpy(y_val, y_pred_raw), 0.8)

            raw_path = write_prediction_parquet(cfg, SampleIndex(subject=1, story=2, split="val"), y_pred_raw, variant="raw")
            smoothed_path = write_prediction_parquet(
                cfg,
                SampleIndex(subject=1, story=2, split="val"),
                y_pred_smoothed,
                variant="smoothed",
            )
            self.assertTrue(raw_path.exists())
            self.assertTrue(smoothed_path.exists())
            self.assertFalse(np.allclose(y_pred_raw, y_pred_smoothed))


if __name__ == "__main__":
    unittest.main()
