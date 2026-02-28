import unittest

from transcript.common import load_config


class TestConfig(unittest.TestCase):
    def test_load_default_config(self):
        cfg = load_config("transcript/config.yaml")
        self.assertIn("paths", cfg)
        self.assertIn("split", cfg)


if __name__ == "__main__":
    unittest.main()
