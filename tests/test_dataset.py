import torch
import numpy as np
from pathlib import Path
from utils.dataset import (
    CharsetManager, OCRDetectionDataset, OCRRecognitionDataset,
    collate_fn, WidthBatchSampler,
)


class TestCharsetManager:
    """Tests for CharsetManager class."""

    def test_blank_token_index_zero(self):
        """Blank token should have index 0."""
        manager = CharsetManager()
        assert manager.char_to_idx["[BLANK]"] == 0

    def test_blank_token_string(self):
        """Blank token string should be [BLANK]."""
        manager = CharsetManager()
        assert manager.blank_token == "[BLANK]"

    def test_persian_chars_present(self):
        """Persian characters should be in charset."""
        manager = CharsetManager()
        for char in "آابپتثجچحخدذرزژسشصضطظعغفقکگلمنوهی":
            assert char in manager.char_to_idx, f"Persian char '{char}' not found"

    def test_english_chars_present(self):
        """English characters should be in charset."""
        manager = CharsetManager()
        for char in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ":
            assert char in manager.char_to_idx, f"English char '{char}' not found"

    def test_digits_present(self):
        """Digits should be in charset."""
        manager = CharsetManager()
        for char in "0123456789۰۱۲۳۴۵۶۷۸۹":
            assert char in manager.char_to_idx, f"Digit '{char}' not found"

    def test_idx_to_char_mapping(self):
        """idx_to_char should be reverse of char_to_idx."""
        manager = CharsetManager()
        for char, idx in manager.char_to_idx.items():
            assert manager.idx_to_char[idx] == char

    def test_num_classes(self):
        """num_classes should match length of char_to_idx."""
        manager = CharsetManager()
        assert manager.num_classes == len(manager.char_to_idx)

    def test_num_classes_minimum(self):
        """Should have at least 50 classes."""
        manager = CharsetManager()
        assert manager.num_classes > 50

    def test_special_chars_present(self):
        """Special characters should be in charset."""
        manager = CharsetManager()
        assert " " in manager.char_to_idx
        assert "." in manager.char_to_idx
        assert "," in manager.char_to_idx

    def test_persian_special_chars(self):
        """Persian special characters should be in charset."""
        manager = CharsetManager()
        for char in "،؛؟»«":
            assert char in manager.char_to_idx, f"Persian special '{char}' not found"


class TestOCRDetectionDataset:
    """Tests for OCRDetectionDataset class."""

    def test_empty_dir_returns_zero_length(self, empty_data_dir):
        """Empty directory should return length 0."""
        dataset = OCRDetectionDataset(empty_data_dir)
        assert len(dataset) == 0


class TestOCRRecognitionDataset:
    """Tests for OCRRecognitionDataset class."""

    def test_empty_dir_returns_zero_length(self, empty_data_dir):
        """Empty directory (no labels.txt) should return length 0."""
        dataset = OCRRecognitionDataset(empty_data_dir, CharsetManager())
        assert len(dataset) == 0, f"Expected 0 samples, got {len(dataset)}"

    def test_with_labels_file(self, empty_data_dir):
        """Should load samples from labels.txt."""
        # Create labels.txt
        labels_file = empty_data_dir / "labels.txt"
        labels_file.write_text("img1.png\tسلام\nimg2.png\thello\n", encoding="utf-8")

        dataset = OCRRecognitionDataset(empty_data_dir, CharsetManager())
        assert len(dataset) == 2

    def test_getitem_empty_dataset(self, empty_data_dir):
        """getitem on empty dataset should return dummy tensor."""
        dataset = OCRRecognitionDataset(empty_data_dir, CharsetManager())
        img, target, length = dataset[0]
        assert img.shape == (1, 32, 128)
        assert target.item() == 0
        assert length == 1

    def test_getitem_with_samples(self, empty_data_dir):
        """getitem should return encoded text."""
        labels_file = empty_data_dir / "labels.txt"
        labels_file.write_text("img1.png\thello\n", encoding="utf-8")

        charset = CharsetManager()
        dataset = OCRRecognitionDataset(empty_data_dir, charset)
        img, target, length = dataset[0]

        assert img.shape == (1, 32, 128)
        assert length == 5  # "hello" has 5 chars
        assert target.shape[0] == 5


class TestWidthBatchSampler:
    """Tests for the width-bucketed batch sampler."""

    def test_length(self):
        sampler = WidthBatchSampler([100] * 100, batch_size=32)
        assert len(sampler) == 100

    def test_ordered_batches_are_width_sorted(self):
        widths = list(range(400, 0, -1))  # descending input
        sampler = WidthBatchSampler(widths, batch_size=32, shuffle=False)
        idxs = list(sampler)
        # Widths of consecutive indices are (mostly) increasing
        ws = [widths[i] for i in idxs]
        assert ws == sorted(ws)

    def test_shuffles_within_chunks(self):
        widths = list(range(1000))
        sampler = WidthBatchSampler(widths, batch_size=32, shuffle=True)
        idxs = list(sampler)
        assert sorted(idxs) == list(range(1000))
        assert idxs != list(range(1000))  # order changed

    def test_each_batch_is_width_homogeneous(self):
        widths = list(range(2000))
        sampler = WidthBatchSampler(widths, batch_size=32, shuffle=False)
        idxs = list(sampler)
        for start in range(0, len(idxs) - 31, 32):
            batch = [widths[i] for i in idxs[start:start + 32]]
            assert max(batch) - min(batch) <= 32  # width spread within one batch


class TestCollateFn:
    """Tests for collate_fn function."""

    def test_collate_single_sample(self):
        """Should handle single sample."""
        img = torch.randn(1, 32, 128)
        target = torch.tensor([1, 2, 3])
        length = torch.tensor(3)

        images, targets, lengths = collate_fn([(img, target, length)])
        assert images.shape == (1, 1, 32, 128)
        assert targets.tolist() == [1, 2, 3]
        assert lengths.tolist() == [3]

    def test_collate_multiple_samples(self):
        """Should handle multiple samples with different lengths."""
        batch = [
            (torch.randn(1, 32, 128), torch.tensor([1, 2, 3]), torch.tensor(3)),
            (torch.randn(1, 32, 128), torch.tensor([4, 5]), torch.tensor(2)),
        ]
        images, targets, lengths = collate_fn(batch)
        assert images.shape == (2, 1, 32, 128)
        assert targets.tolist() == [1, 2, 3, 4, 5]
        assert lengths.tolist() == [3, 2]
