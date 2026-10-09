import numpy as np
import cv2
import tempfile
from pathlib import Path
from inference import OCREngine


class TestOCREngine:
    """Tests for OCREngine class."""

    def test_engine_init(self):
        """Engine should initialize without error."""
        engine = OCREngine()
        assert engine is not None

    def test_engine_has_device(self):
        """Engine should have a device set."""
        engine = OCREngine()
        assert engine.device is not None

    def test_engine_has_config(self):
        """Engine should load config."""
        engine = OCREngine()
        assert engine.config is not None
        assert 'ocr' in engine.config

    def test_engine_has_backend(self):
        """Engine should have a backend initialized."""
        engine = OCREngine()
        assert engine.backend is not None

    def test_engine_has_preprocessor(self):
        """Engine should have a preprocessor."""
        engine = OCREngine()
        assert engine.preprocessor is not None

    def test_engine_has_pdf_processor(self):
        """Engine should have a PDF processor."""
        engine = OCREngine()
        assert engine.pdf_processor is not None

    def test_read_image_valid(self, sample_image_path):
        """Should read text from valid image path."""
        engine = OCREngine()
        result = engine.read_image(sample_image_path)
        assert "text" in result
        assert "regions" in result
        assert "image_path" in result

    def test_read_image_nonexistent(self):
        """Should handle nonexistent image gracefully."""
        engine = OCREngine()
        result = engine.read_image("nonexistent_image.png")
        assert result["text"] == ""
        assert "error" in result

    def test_read_image_no_double_preprocessing(self, sample_image_path):
        """read_image should NOT apply preprocessing twice (backend handles it)."""
        engine = OCREngine()
        from unittest.mock import patch
        
        with patch.object(engine.backend, 'recognize_with_details') as mock_recognize:
            mock_recognize.return_value = {"text": "test", "confidence": 1.0}
            result = engine.read_image(sample_image_path)
            
            assert mock_recognize.call_count == 1
            call_args = mock_recognize.call_args[0]
            assert isinstance(call_args[0], np.ndarray)

    def test_recognize_numpy_image(self, text_image_bgr):
        """Should recognize text from numpy image."""
        engine = OCREngine()
        result = engine.recognize(text_image_bgr)
        assert "text" in result
        assert "confidence" in result

    def test_recognize_no_double_postprocessing(self, text_image_bgr):
        """recognize should NOT apply postprocessing twice."""
        engine = OCREngine()
        from unittest.mock import patch
        
        with patch.object(engine.backend, 'recognize_with_details') as mock_recognize:
            mock_recognize.return_value = {"text": "test", "confidence": 1.0}
            result = engine.recognize(text_image_bgr)
            
            assert mock_recognize.call_count == 1

    def test_read_pdf(self, tmp_path):
        """Should handle PDF reading (even if PDF is invalid)."""
        pdf_path = tmp_path / "test.pdf"
        pdf_path.write_bytes(b"%PDF-1.4\n1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj\n3 0 obj<</Type/Page/MediaBox[0 0 612 792]/Parent 2 0 R>>endobj\nxref\n0 4\n0000000000 65535 f \n0000000009 00000 n \n0000000058 00000 n \n0000000115 00000 n \ntrailer<</Size 4/Root 1 0 R>>\nstartxref\n190\n%%EOF")
        
        engine = OCREngine()
        result = engine.read(str(pdf_path))
        assert "pages" in result
        assert "filename" in result

    def test_read_unsupported_format(self, tmp_path):
        """Should raise ValueError for unsupported format."""
        unsupported = tmp_path / "test.xyz"
        unsupported.write_text("test")
        
        engine = OCREngine()
        try:
            engine.read(str(unsupported))
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "Unsupported" in str(e)

    def test_crop_region_valid(self):
        """Should crop image region correctly."""
        engine = OCREngine()
        img = np.zeros((100, 200), dtype=np.uint8)
        img[10:50, 20:100] = 255
        
        cropped = engine._crop_region(img, [20, 10, 100, 50])
        assert cropped.shape == (40, 80)

    def test_crop_region_out_of_bounds(self):
        """Should clamp out-of-bounds coordinates."""
        engine = OCREngine()
        img = np.zeros((100, 200), dtype=np.uint8)
        
        cropped = engine._crop_region(img, [150, 80, 300, 200])
        assert cropped.shape[0] > 0
        assert cropped.shape[1] > 0

    def test_crop_region_invalid(self):
        """Should return empty image for invalid box."""
        engine = OCREngine()
        img = np.zeros((100, 200), dtype=np.uint8)
        
        cropped = engine._crop_region(img, [50, 50, 30, 30])
        assert cropped.shape == (32, 128)

    def test_crop_region_exception_returns_image(self):
        """Should return original image on exception."""
        engine = OCREngine()
        img = np.zeros((100, 200), dtype=np.uint8)
        # Pass a box that will cause an error in int() conversion
        cropped = engine._crop_region(img, [None, None, None, None])
        # On exception, returns original image
        assert cropped is img

    def test_read_image_returns_dict(self, sample_image_path):
        """read_image should always return a dict."""
        engine = OCREngine()
        result = engine.read_image(sample_image_path)
        assert isinstance(result, dict)

    def test_recognize_returns_dict(self, text_image_bgr):
        """recognize should always return a dict."""
        engine = OCREngine()
        result = engine.recognize(text_image_bgr)
        assert isinstance(result, dict)
