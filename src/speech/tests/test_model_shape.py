import unittest

import torch

from speech.common import SpeechBiGRUModel, load_config


class TestSpeechModelShape(unittest.TestCase):
    def test_forward_shape_and_batch_norm(self):
        cfg = load_config("speech/config.yaml")
        model = SpeechBiGRUModel(cfg)
        self.assertTrue(hasattr(model, "batch_norm"))

        bsz = 3
        seq_len = int(cfg["model"]["sequence_length"])
        feat_dim = int(cfg["feature"]["n_freq_bins"])
        x = torch.randn(bsz, seq_len, feat_dim, dtype=torch.float32)
        sid = torch.tensor([0, 1, 2], dtype=torch.long)

        y = model(x, sid)
        self.assertEqual(tuple(y.shape), (bsz, seq_len))


if __name__ == "__main__":
    unittest.main()
