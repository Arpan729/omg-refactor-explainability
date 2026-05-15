import os
from pathlib import Path
import tempfile
import unittest

from late_fusion.common import load_config


class TestConfig(unittest.TestCase):
    def test_load_minimal_temp_config(self):
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            for rel in ["train_ann", "val_ann", "speech_pred", "raw_pred"]:
                (td_path / rel).mkdir(parents=True, exist_ok=True)

            cfg_path = td_path / "config.yaml"
            cfg_path.write_text(
                """
paths:
  train_ann_dir: train_ann
  val_ann_dir: val_ann
  prediction_dir: out/predictions
split:
  manifest_id: test
  subjects_train: [1]
  subjects_val: [1]
  stories_train: [1]
  stories_val: [2]
fusion:
  fps: 25.0
  final_cutoff: 0.01
  final_order: 1
  modalities:
    speech:
      prediction_dir: speech_pred
      kind: frame
      weight: 1.0
      cutoff: 0.004
      order: 1
    raw_face:
      prediction_dir: raw_pred
      kind: frame
      weight: 0.1
      cutoff: 0.006
      order: 1
predict:
  device: cpu
                """.strip(),
                encoding="utf-8",
            )

            old = Path.cwd()
            try:
                os.chdir(td)
                cfg = load_config(str(cfg_path))
                self.assertIn("paths", cfg)
                self.assertIn("fusion", cfg)
            finally:
                os.chdir(old)


if __name__ == "__main__":
    unittest.main()
