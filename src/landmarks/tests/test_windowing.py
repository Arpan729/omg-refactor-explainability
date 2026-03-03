import unittest

import numpy as np

from landmarks.common import window_landmarks


class TestWindowing(unittest.TestCase):
    def test_window_shape(self):
        x = np.zeros((10, 136), dtype=np.float32)
        xw = window_landmarks(x, window_size=5)
        self.assertEqual(xw.shape, (10, 5, 136))


if __name__ == "__main__":
    unittest.main()
