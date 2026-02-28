import unittest
import numpy as np

from transcript_next.common import window_sequence


class TestWindowing(unittest.TestCase):
    def test_window_count(self):
        x = np.zeros((10, 11), dtype=np.float32)
        y = np.arange(10, dtype=np.float32)
        xw, yw = window_sequence(x, y, window_size=4, stride=2)
        self.assertEqual(len(xw), 4)
        self.assertEqual(len(yw), 4)


if __name__ == "__main__":
    unittest.main()
