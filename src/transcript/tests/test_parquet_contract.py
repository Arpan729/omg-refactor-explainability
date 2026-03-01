import tempfile
import unittest
from pathlib import Path

import numpy as np

from transcript.common import SampleIndex, validate_prediction_parquet, write_prediction_parquet


class TestParquetContract(unittest.TestCase):
    def test_window_level_columns_exist(self):
        with tempfile.TemporaryDirectory() as td:
            cfg = {
                "paths": {"prediction_dir": td},
                "model": {"window_size": 100, "stride": 50},
                "split": {"manifest_id": "test_v1"},
            }
            sample = SampleIndex(subject=1, story=1, split="val")
            y_pred = np.array([0.1, 0.2, 0.3], dtype=np.float32)
            out = write_prediction_parquet(cfg, sample, y_pred)
            validate_prediction_parquet(out)
            self.assertTrue(Path(out).exists())


if __name__ == "__main__":
    unittest.main()
