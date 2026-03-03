import tempfile
import unittest
from pathlib import Path
import os

from landmarks.common import load_config


class TestConfig(unittest.TestCase):
    def test_load_minimal_temp_config(self):
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            for rel in ["train_ann", "val_ann", "train_lm", "val_lm"]:
                (td_path / rel).mkdir(parents=True, exist_ok=True)
            for rel in ["train_vid", "val_vid"]:
                (td_path / rel).mkdir(parents=True, exist_ok=True)
            (td_path / "predictor.dat").write_bytes(b"stub")

            cfg_path = td_path / "config.yaml"
            cfg_path.write_text(
                """
paths:
  train_videos_dir: train_vid
  val_videos_dir: val_vid
  predictor_path: predictor.dat
  train_ann_dir: train_ann
  val_ann_dir: val_ann
  train_landmarks_csv_dir: train_lm
  val_landmarks_csv_dir: val_lm
  feature_dir: out/features
  checkpoint_dir: out/checkpoints
  prediction_dir: out/predictions
split:
  manifest_id: test
  subjects_train: [1]
  subjects_val: [1]
  stories_train: [2]
  stories_val: [1]
audio:
  fps: 25.0
extract:
  detector_upsample: 1
model:
  window_size: 5
  conv_channels: [100,100,160,160]
  kernel_size: 2
  dense_dim: 32
  dropout: 0.0
train:
  epochs: 1
  batch_size: 1
  lr: 0.001
  weight_decay: 0.0
  patience: 1
  seed: 42
  device: cpu
predict:
  batch_size: 1
  device: cpu
                """.strip(),
                encoding="utf-8",
            )

            old = Path.cwd()
            try:
                # path resolver uses cwd semantics by design
                os.chdir(td)
                cfg = load_config(str(cfg_path))
                self.assertIn("paths", cfg)
                self.assertIn("split", cfg)
            finally:
                os.chdir(old)


if __name__ == "__main__":
    unittest.main()
