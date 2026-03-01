import unittest

from speech.common import load_config


class TestConfig(unittest.TestCase):
    def test_load_default_config(self):
        cfg = load_config("speech/config.yaml")
        self.assertIn("paths", cfg)
        self.assertIn("split", cfg)
        self.assertIn("highpass_cutoff_hz", cfg["feature"])
        self.assertIn("highpass_order", cfg["feature"])
        self.assertIn("preemphasis_mode", cfg["feature"])
        self.assertIn("compression_type", cfg["feature"])
        self.assertIn("compression_power", cfg["feature"])
        self.assertIn("use_batch_norm", cfg["model"])


if __name__ == "__main__":
    unittest.main()
