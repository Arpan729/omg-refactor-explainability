import tempfile
import unittest
from pathlib import Path
import os

from raw_face.common import load_config


class TestConfig(unittest.TestCase):
    def test_load_minimal_temp_config(self):
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            for rel in ["train_vid", "val_vid", "train_ann", "val_ann"]:
                (td_path / rel).mkdir(parents=True, exist_ok=True)

            cfg_path = td_path / "config.yaml"
            cfg_path.write_text(
                """
paths:
  train_videos_dir: train_vid
  val_videos_dir: val_vid
  train_ann_dir: train_ann
  val_ann_dir: val_ann
  train_faces_dir: out/faces/train
  val_faces_dir: out/faces/val
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
  image_ext: ".png"
  face_size: 48
model:
  seq_len: 10
  conv_channels: [32,64]
  dense_dim: 128
  subject_embed_dim: 3
  dropout: 0.6
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
                os.chdir(td)
                cfg = load_config(str(cfg_path))
                self.assertIn("paths", cfg)
                self.assertIn("split", cfg)
            finally:
                os.chdir(old)


if __name__ == "__main__":
    unittest.main()
