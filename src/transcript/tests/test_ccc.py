import unittest
import numpy as np

from transcript.common import ccc_numpy


class TestCCC(unittest.TestCase):
    def test_perfect(self):
        y = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        self.assertGreater(ccc_numpy(y, y), 0.9999)

    def test_constant_no_nan(self):
        a = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        b = np.array([0.5, 0.5, 0.5], dtype=np.float32)
        self.assertTrue(np.isfinite(ccc_numpy(a, b)))


if __name__ == "__main__":
    unittest.main()
