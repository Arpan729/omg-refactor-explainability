import os
from pathlib import Path
import tempfile
import unittest

from ensemble_fusion.common import load_config


class TestConfig(unittest.TestCase):
    def test_load_minimal_temp_config(self):
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            for rel in ["train_ann", "val_ann", "speech_pred", "speech_oof"]:
                (td_path / rel).mkdir(parents=True, exist_ok=True)

            cfg_path = td_path / "config.yaml"
            cfg_path.write_text(
                """
paths:
  train_ann_dir: train_ann
  val_ann_dir: val_ann
  checkpoint_dir: out/checkpoints
  prediction_dir_raw: out/predictions_raw
  prediction_dir_smoothed: out/predictions_smoothed
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
      oof_dir: speech_oof
      kind: frame
      cutoff: 0.004
      order: 1
model:
  positive: true
  fit_intercept: true
  max_iter: 1000
train:
  mode: sweep
  model_name: elastic_net_positive
  selection_variant: raw
  model_names: [ridge_positive, elastic_net_positive]
  ridge_alpha_grid: [0.0, 0.001]
  elastic_net_alpha_grid: [0.0001]
  elastic_net_l1_ratio_grid: [0.1, 0.5]
  alpha: 0.001
  l1_ratio: 0.5
  seed: 42
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
                self.assertIn("prediction_dir_raw", cfg["paths"])
                self.assertIn("prediction_dir_smoothed", cfg["paths"])
                self.assertEqual(cfg["train"]["mode"], "sweep")
            finally:
                os.chdir(old)


if __name__ == "__main__":
    unittest.main()
