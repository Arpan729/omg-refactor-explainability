import unittest

import numpy as np

from speech.common import apply_highpass_filter, apply_preemphasis


class TestSpeechDSP(unittest.TestCase):
    def test_highpass_filter_preserves_length(self):
        sr = 16000
        t = np.arange(sr, dtype=np.float32) / sr
        x = np.sin(2.0 * np.pi * 50.0 * t).astype(np.float32)
        y = apply_highpass_filter(x, sample_rate=sr, cutoff_hz=100.0, order=8)
        self.assertEqual(len(x), len(y))
        self.assertTrue(np.all(np.isfinite(y)))

    def test_preemphasis_paper_equation(self):
        x = np.array([3.0, 6.0, 9.0, 12.0], dtype=np.float32)
        y = apply_preemphasis(x, mode="paper_eq1")
        expected = np.array([1.0, 3.0, 4.0, 5.0], dtype=np.float32)
        np.testing.assert_allclose(y, expected, rtol=1e-6, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
