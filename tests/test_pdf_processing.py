from preprocessing.pdf_processing import PDFProcessor

def test_pdf_processor_init():
    class MockEngine:
        def recognize(self, img):
            return {"text": "تست", "confidence": 1.0}
            
    processor = PDFProcessor(ocr_engine=MockEngine())
    assert processor.ocr_engine is not None
