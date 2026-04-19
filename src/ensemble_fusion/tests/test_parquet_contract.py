import tempfile
import unittest

import numpy as np

from ensemble_fusion.common import SampleIndex, validate_prediction_parquet, write_prediction_parquet


class TestParquetContract(unittest.TestCase):
    def test_frame_level_columns_exist(self):
        with tempfile.TemporaryDirectory() as td:
            cfg = {
                "paths": {"prediction_dir_raw": td, "prediction_dir_smoothed": td},
                "fusion": {"fps": 25.0},
                "split": {"manifest_id": "ensemble_fusion_test_v1"},
            }
            sample = SampleIndex(subject=1, story=1, split="val")
            y_pred = np.array([0.1, 0.2, 0.3], dtype=np.float32)
            out = write_prediction_parquet(cfg, sample, y_pred, variant="raw")
            validate_prediction_parquet(out)


if __name__ == "__main__":
    unittest.main()
