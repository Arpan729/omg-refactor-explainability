import unittest

import numpy as np

from speech.common import reconstruct_from_windows, window_features, window_sequence


class TestWindowing(unittest.TestCase):
    def test_window_count_and_reconstruct_length(self):
        x = np.zeros((10, 5), dtype=np.float32)
        y = np.arange(10, dtype=np.float32)
        xw, yw, starts = window_sequence(x, y, window_size=4, stride=3, include_last=True)
        self.assertEqual(len(xw), 3)
        self.assertEqual(len(yw), 3)
        pred = np.asarray(yw, dtype=np.float32)
        rec = reconstruct_from_windows(pred, starts, total_len=10)
        self.assertEqual(len(rec), 10)

    def test_feature_only_windowing(self):
        x = np.arange(50, dtype=np.float32).reshape(10, 5)
        xw, starts = window_features(x, window_size=4, stride=3, include_last=True)
        self.assertEqual(xw.shape, (3, 4, 5))
        self.assertEqual(starts.tolist(), [0, 3, 6])


if __name__ == "__main__":
    unittest.main()
