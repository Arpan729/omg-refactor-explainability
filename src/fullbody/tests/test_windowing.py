import unittest

import numpy as np

from fullbody.common import window_sequence_legacy


class TestWindowing(unittest.TestCase):
    def test_window_shape(self):
        x = np.zeros((20, 1, 48, 48), dtype=np.float32)
        y = np.arange(20, dtype=np.float32)
        xw, yw, frame_idx = window_sequence_legacy(x, y, seq_len=5)

        self.assertEqual(xw.shape, (15, 1, 5, 48, 48))
        self.assertEqual(yw.shape, (15, 1))
        self.assertEqual(frame_idx.shape, (15,))


if __name__ == "__main__":
    unittest.main()
