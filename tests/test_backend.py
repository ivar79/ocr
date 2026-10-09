import numpy as np
import cv2
from unittest.mock import patch, MagicMock
from backend.ocr_backend import TesseractOCRBackend, OCRBackendFactory


class TestTesseractOCRBackend:
    """Tests for TesseractOCRBackend class."""

    def test_init_default_language(self):
        """Default language should be fas+eng."""
        backend = TesseractOCRBackend()
        assert backend.language == "fas+eng"

    def test_init_custom_language(self):
        """Custom language should be set."""
        backend = TesseractOCRBackend(language="eng")
        assert backend.language == "eng"

    def test_candidate_tessdata_dirs_not_empty(self):
        """Should return at least one candidate directory."""
        dirs = TesseractOCRBackend._candidate_tessdata_dirs()
        assert len(dirs) > 0

    def test_candidate_tessdata_dirs_unique(self):
        """Should return unique directories."""
        dirs = TesseractOCRBackend._candidate_tessdata_dirs()
        resolved = [str(d.resolve(strict=False)) for d in dirs]
        assert len(resolved) == len(set(resolved))

    def test_load_image_from_numpy(self):
        """Should accept numpy array directly."""
        backend = TesseractOCRBackend()
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        result = backend._load_image(img)
        assert result.shape == (100, 100, 3)

    def test_load_image_none_raises(self):
        """Should raise ValueError for None image."""
        backend = TesseractOCRBackend()
        try:
            backend._load_image(None)
            assert False, "Should have raised ValueError"
        except ValueError:
            pass

    def test_get_config_default(self):
        """Default config should include --oem 3."""
        backend = TesseractOCRBackend()
        config = backend._get_config()
        assert "--oem 3" in config
        assert "--psm" in config

    def test_get_config_custom_psm(self):
        """Custom PSM should be included."""
        backend = TesseractOCRBackend()
        config = backend._get_config(psm=3)
        assert "--psm 3" in config

    def test_psm_modes_defined(self):
        """PSM_MODES should be defined."""
        assert hasattr(TesseractOCRBackend, 'PSM_MODES')
        assert 'single_block' in TesseractOCRBackend.PSM_MODES
        assert 'auto' in TesseractOCRBackend.PSM_MODES

    def test_analyze_image(self):
        """Should analyze image and return recommendations."""
        backend = TesseractOCRBackend()
        img = np.zeros((200, 400), dtype=np.uint8)
        img[50:100, 50:350] = 255
        analysis = backend._analyze_image(img)
        assert 'width' in analysis
        assert 'height' in analysis
        assert 'recommended_psm' in analysis

    def test_ensure_language_data_cache(self):
        """Should cache language check."""
        TesseractOCRBackend._fas_checked = False
        TesseractOCRBackend._ensure_language_data("eng")
        TesseractOCRBackend._ensure_language_data("eng")

    def test_ensure_language_data_empty(self):
        """Empty language string should return without error."""
        TesseractOCRBackend._ensure_language_data("")

    def test_preprocessor_initialized(self):
        """Preprocessor should be initialized."""
        backend = TesseractOCRBackend()
        assert backend.preprocessor is not None

    def test_postprocessor_initialized(self):
        """Postprocessor should be initialized."""
        backend = TesseractOCRBackend()
        assert backend.postprocessor is not None

    def test_enable_retry_default(self):
        """enable_retry should be True by default."""
        backend = TesseractOCRBackend()
        assert backend.enable_retry is True

    def test_confidence_threshold_default(self):
        """confidence_threshold should be 0.5 by default."""
        backend = TesseractOCRBackend()
        assert backend.confidence_threshold == 0.5

    def test_confidence_threshold_custom(self):
        """Custom confidence_threshold should be set."""
        backend = TesseractOCRBackend(confidence_threshold=0.7)
        assert backend.confidence_threshold == 0.7

    def test_max_retries_default(self):
        """max_retries should default to 2."""
        backend = TesseractOCRBackend()
        assert backend.max_retries == 2

    def test_preprocessing_config_passthrough(self):
        """Preprocessing config should reach the preprocessor."""
        backend = TesseractOCRBackend(
            preprocessing_config={'mode': 'fast', 'upscale_factor': 3})
        assert backend.preprocessor.mode == 'fast'
        assert backend.preprocessor.upscale_factor == 3


class TestRebuildText:
    """Tests for multi-line text reconstruction from image_to_data."""

    @staticmethod
    def _make_data(lines):
        """Build a synthetic image_to_data dict from a list of word lists."""
        text, conf, block_num, par_num, line_num = [], [], [], [], []
        for b, (line_words, confs) in enumerate(lines):
            for w, c in zip(line_words, confs):
                text.append(w)
                conf.append(c)
                block_num.append(0)
                par_num.append(0)
                line_num.append(b)
        return {
            'text': text, 'conf': conf, 'block_num': block_num,
            'par_num': par_num, 'line_num': line_num,
        }

    def test_single_line(self):
        data = self._make_data([(["Hello", "World"], [90, 95])])
        assert TesseractOCRBackend._rebuild_text(data) == "Hello World"

    def test_multiple_lines(self):
        data = self._make_data([
            (["سلام", "دنیا"], [90, 95]),
            (["Hello", "World"], [88, 92]),
        ])
        assert TesseractOCRBackend._rebuild_text(data) == "سلام دنیا\nHello World"

    def test_empty_words_skipped(self):
        data = self._make_data([(["", "Hello", " ", "World"], [-1, 90, -1, 95])])
        assert TesseractOCRBackend._rebuild_text(data) == "Hello World"

    def test_empty_data(self):
        data = {'text': [], 'conf': [], 'block_num': [], 'par_num': [], 'line_num': []}
        assert TesseractOCRBackend._rebuild_text(data) == ""

    def test_ocr_with_confidence_single_call(self):
        """One image_to_data call provides both text and confidence."""
        backend = TesseractOCRBackend()
        img = np.zeros((100, 300), dtype=np.uint8)
        fake_data = {
            'text': ['Hello', 'World'], 'conf': [90, 95],
            'block_num': [0, 0], 'par_num': [0, 0], 'line_num': [0, 0],
        }
        with patch('backend.ocr_backend.pytesseract.image_to_data',
                   return_value=fake_data) as mock_to_data, \
             patch('backend.ocr_backend.pytesseract.image_to_string') as mock_to_string:
            text, confidence = backend._ocr_with_confidence(img, psm=6)
        mock_to_data.assert_called_once()
        mock_to_string.assert_not_called()
        assert text == "Hello World"
        assert abs(confidence - 0.925) < 1e-6

    def test_recognize_best_keeps_best(self):
        """Retry should keep the highest-confidence non-empty result."""
        backend = TesseractOCRBackend(confidence_threshold=0.9, max_retries=3)
        img = np.zeros((100, 300), dtype=np.uint8)
        results = {
            6: ("poor text", 0.4),
            3: ("better text", 0.6),
            11: ("best text", 0.85),
        }

        def fake_ocr(image, psm):
            return results[psm]

        with patch.object(backend, '_ocr_with_confidence', side_effect=fake_ocr), \
             patch.object(backend, '_analyze_image', return_value={'recommended_psm': 6}):
            text, confidence, psm = backend._recognize_best(img, 6)
        assert text == "best text"
        assert confidence == 0.85
        assert psm == 11

    def test_recognize_best_no_retry_when_confident(self):
        """No retry when the first attempt already meets the threshold."""
        backend = TesseractOCRBackend(confidence_threshold=0.5, max_retries=2)
        img = np.zeros((100, 300), dtype=np.uint8)

        with patch.object(backend, '_ocr_with_confidence',
                          return_value=("good text", 0.8)) as mock_ocr, \
             patch.object(backend, '_analyze_image', return_value={'recommended_psm': 6}):
            text, confidence, psm = backend._recognize_best(img, 6)
        mock_ocr.assert_called_once()
        assert text == "good text"

    def test_recognize_best_ignores_empty_alternatives(self):
        """An empty alternative should not replace a non-empty result."""
        backend = TesseractOCRBackend(confidence_threshold=0.9, max_retries=3)
        img = np.zeros((100, 300), dtype=np.uint8)
        results = {6: ("initial", 0.3), 3: ("", 0.99), 11: ("better", 0.6)}

        def fake_ocr(image, psm):
            return results[psm]

        with patch.object(backend, '_ocr_with_confidence', side_effect=fake_ocr), \
             patch.object(backend, '_analyze_image', return_value={'recommended_psm': 6}):
            text, confidence, psm = backend._recognize_best(img, 6)
        assert text == "better"
        assert psm == 11


class TestOCRBackendFactory:
    """Tests for OCRBackendFactory class."""

    def test_create_tesseract(self):
        """Should create TesseractOCRBackend for 'tesseract' backend."""
        backend = OCRBackendFactory.create(backend="tesseract", language="fas+eng")
        assert isinstance(backend, TesseractOCRBackend)

    def test_create_tesseract_case_insensitive(self):
        """Should handle case-insensitive backend name."""
        backend = OCRBackendFactory.create(backend="Tesseract", language="fas+eng")
        assert isinstance(backend, TesseractOCRBackend)

    def test_create_unsupported_raises(self):
        """Should raise ValueError for unsupported backend."""
        try:
            OCRBackendFactory.create(backend="unsupported")
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "Unsupported" in str(e)

    def test_create_default_language(self):
        """Should use default language when not specified."""
        backend = OCRBackendFactory.create(backend="tesseract")
        assert backend.language == "fas+eng"
