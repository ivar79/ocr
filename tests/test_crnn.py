import torch

from models.recognition.crnn import CRNN
from utils.dataset import CharsetManager


class TestCRNNCheckpoint:
    """Tests for CRNN architecture persistence."""

    def test_arch_metadata(self):
        model = CRNN(32, 1, 100, hidden_size=128, cnn_scale=0.5)
        arch = model.arch()
        assert arch["hidden_size"] == 128
        assert arch["cnn_scale"] == 0.5
        assert arch["num_classes"] == 100

    def test_save_load_round_trip(self, tmp_path):
        path = str(tmp_path / "model.pt")
        model = CRNN(32, 1, 100, hidden_size=128, cnn_scale=0.5)
        model.save_checkpoint(path)

        loaded = CRNN.from_checkpoint(path)
        assert loaded.hidden_size == 128
        assert loaded.cnn_scale == 0.5
        assert loaded.num_classes == 100
        # Weights should round-trip exactly
        for (k1, v1), (k2, v2) in zip(
                model.state_dict().items(), loaded.state_dict().items()):
            assert k1 == k2
            assert torch.equal(v1, v2)

    def test_load_plain_state_dict(self, tmp_path):
        """Legacy plain state dicts still load (classes inferred)."""
        path = str(tmp_path / "plain.pt")
        model = CRNN(32, 1, 123)
        torch.save(model.state_dict(), path)

        loaded = CRNN.from_checkpoint(path)
        assert loaded.num_classes == 123
        assert loaded.hidden_size == 256  # default full architecture

    def test_forward_output_shape(self):
        """Forward should produce (B, T, num_classes)."""
        charset = CharsetManager()
        model = CRNN(32, 1, charset.num_classes, hidden_size=64, cnn_scale=0.5)
        x = torch.randn(2, 1, 32, 128)
        out = model(x)
        assert out.dim() == 3
        assert out.size(0) == 2
        assert out.size(2) == charset.num_classes

    def test_ctc_decoder_no_crash(self):
        charset = CharsetManager()
        model = CRNN(32, 1, charset.num_classes, hidden_size=64, cnn_scale=0.5)
        x = torch.randn(1, 1, 32, 128)
        preds = model(x)
        text = model.ctc_greedy_decoder(preds)
        assert isinstance(text, str)
