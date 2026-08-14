import fitz  # PyMuPDF
from pathlib import Path

class PDFProcessor:
    def __init__(self, ocr_engine):
        self.ocr_engine = ocr_engine
    
    def process(self, pdf_path: str) -> dict:
        """پردازش PDF و استخراج متن"""
        pdf_path = Path(pdf_path)
        doc = fitz.open(str(pdf_path))
        
        results = {
            "filename": pdf_path.name,
            "total_pages": len(doc),
            "pages": []
        }
        
        for page_num, page in enumerate(doc):
            # ابتدا تلاش برای استخراج متن مستقیم
            text = page.get_text("text").strip()
            
            if text:
                # PDF دیجیتال — متن مستقیماً قابل استخراج
                results["pages"].append({
                    "page": page_num + 1,
                    "type": "digital",
                    "text": text,
                    "confidence": 1.0
                })
            else:
                # PDF اسکن‌شده — نیاز به OCR
                pix = page.get_pixmap(matrix=fitz.Matrix(300/72, 300/72))
                img = self._pixmap_to_numpy(pix)
                
                ocr_result = self.ocr_engine.recognize(img)
                results["pages"].append({
                    "page": page_num + 1,
                    "type": "scanned",
                    "text": ocr_result["text"],
                    "confidence": ocr_result["confidence"]
                })
        
        doc.close()
        return results
    
    def _pixmap_to_numpy(self, pix):
        import numpy as np
        img = np.frombuffer(pix.samples, dtype=np.uint8)
        img = img.reshape(pix.height, pix.width, pix.n)
        return img
