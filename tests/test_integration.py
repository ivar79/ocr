import numpy as np
import cv2
from pathlib import Path
from inference import OCREngine


class TestIntegration:
    """Integration tests for the OCR system."""

    def test_engine_full_init(self):
        """Engine should fully initialize with all components."""
        engine = OCREngine()
        assert engine.device is not None
        assert engine.config is not None
        assert engine.backend is not None
        assert engine.preprocessor is not None
        assert engine.pdf_processor is not None

    def test_engine_read_image_end_to_end(self, sample_image_path):
        """Full pipeline: image file -> text."""
        engine = OCREngine()
        result = engine.read_image(sample_image_path)
        assert isinstance(result, dict)
        assert "text" in result
        assert isinstance(result["text"], str)

    def test_engine_recognize_end_to_end(self, text_image_bgr):
        """Full pipeline: numpy image -> text."""
        engine = OCREngine()
        result = engine.recognize(text_image_bgr)
        assert isinstance(result, dict)
        assert "text" in result
        assert "confidence" in result

    def test_engine_read_dispatches_image(self, sample_image_path):
        """read() should dispatch to read_image for .png files."""
        engine = OCREngine()
        result = engine.read(sample_image_path)
        assert "text" in result

    def test_engine_read_dispatches_pdf(self, tmp_path):
        """read() should dispatch to pdf_processor for .pdf files."""
        pdf_path = tmp_path / "test.pdf"
        # Minimal valid PDF
        pdf_path.write_bytes(b"%PDF-1.4\n1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj\n3 0 obj<</Type/Page/MediaBox[0 0 612 792]/Parent 2 0 R>>endobj\nxref\n0 4\n0000000000 65535 f \n0000000009 00000 n \n0000000058 00000 n \n0000000115 00000 n \ntrailer<</Size 4/Root 1 0 R>>\nstartxref\n190\n%%EOF")
        
        engine = OCREngine()
        result = engine.read(str(pdf_path))
        assert "pages" in result

    def test_engine_batch_images(self, tmp_path):
        """Should process multiple images without error."""
        engine = OCREngine()
        
        # Create multiple test images
        for i in range(3):
            img = np.zeros((50, 100, 3), dtype=np.uint8)
            cv2.putText(img, f"Test {i}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            img_path = tmp_path / f"test_{i}.png"
            cv2.imwrite(str(img_path), img)
        
        results = []
        for img_file in tmp_path.glob("*.png"):
            result = engine.read_image(str(img_file))
            results.append(result)
        
        assert len(results) == 3
        for result in results:
            assert "text" in result

    def test_engine_consistent_output(self, text_image_bgr):
        """Same input should produce same output."""
        engine = OCREngine()
        result1 = engine.recognize(text_image_bgr)
        result2 = engine.recognize(text_image_bgr)
        assert result1["text"] == result2["text"]

    def test_engine_config_loading(self):
        """Engine should load config from config.yaml."""
        engine = OCREngine()
        assert engine.config["ocr"]["backend"] == "tesseract"
        assert engine.config["ocr"]["language"] == "fas+eng"

    def test_engine_backend_language(self):
        """Backend should use configured language."""
        engine = OCREngine()
        assert engine.backend.language == "fas+eng"
