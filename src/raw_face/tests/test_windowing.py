import unittest

import numpy as np

from raw_face.common import window_sequence


class TestWindowing(unittest.TestCase):
    def test_window_shape(self):
        x = np.zeros((10, 1, 48, 48), dtype=np.float32)
        y = np.arange(10, dtype=np.float32)
        sid = np.zeros((10,), dtype=np.int64)
        xw, yw, sw = window_sequence(x, y, sid, seq_len=5)
        self.assertEqual(xw.shape, (10, 1, 5, 48, 48))
        self.assertEqual(yw.shape, (10, 1))
        self.assertEqual(sw.shape, (10,))


if __name__ == "__main__":
    unittest.main()
