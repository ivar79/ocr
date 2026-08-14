import torch
import cv2
import yaml
import logging
import numpy as np
import os
import pytesseract
from pathlib import Path

from preprocessing.image_processing import ImagePreprocessor
from preprocessing.pdf_processing import PDFProcessor
from postprocessing.text_formatter import TextPostProcessor
from backend.ocr_backend import OCRBackendFactory

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# Configure Tesseract path
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
os.environ['TESSDATA_PREFIX'] = str(Path.home() / 'AppData' / 'Local' / 'Tesseract-OCR' / 'tessdata')

class DummyDetector:
    def detect(self, image): 
        logging.warning("Using DummyDetector. No text regions will be found.")
        return []

class DummyRecognizer:
    def __init__(self): self.last_confidence = 1.0
    def recognize(self, image): 
        logging.warning("Using DummyRecognizer. Returning empty text.")
        return ""

class OCREngine:
    def __init__(self, config_path: str = "configs/config.yaml"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logging.info(f"Using device: {self.device}")

        self.config = self._load_config(config_path)

        self.preprocessor = ImagePreprocessor(
            target_size=tuple(self.config.get('preprocessing', {}).get('target_size', [640, 640]))
        )
        self.postprocessor = TextPostProcessor()
        self.pdf_processor = PDFProcessor(self)

        backend_name = self.config.get('ocr', {}).get('backend', 'tesseract')
        language = self.config.get('ocr', {}).get('language', 'fas+eng')
        self.backend = OCRBackendFactory.create(backend=backend_name, language=language)
    
    def _load_config(self, config_path):
        default_config = {
            'ocr': {'backend': 'tesseract', 'language': 'fas+eng'},
            'detection': {'weights': 'models/pretrained/craft_weights.pth'},
            'recognition': {'weights': 'models/pretrained/crnn_weights.pth'},
            'preprocessing': {'target_size': [640, 640]}
        }
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
                if config is None:
                    config = {}
                for k, v in config.items():
                    if isinstance(v, dict) and k in default_config:
                        default_config[k].update(v)
                    else:
                        default_config[k] = v
                return default_config
        except Exception as e:
            logging.warning(f"Could not load config from {config_path}: {e}. Using defaults.")
            return default_config

    def _load_detector(self):
        logging.info("Using OCR backend instead of CRAFT detector.")
        return None

    def _load_recognizer(self):
        logging.info("Using OCR backend instead of CRNN recognizer.")
        return None
        
    def _crop_region(self, image, box):
        """Crop the image given a bounding box [x1, y1, x2, y2]"""
        try:
            h, w = image.shape[:2]
            x1, y1, x2, y2 = map(int, box)
            
            # Clamp to image dimensions
            x1 = max(0, min(x1, w-1))
            y1 = max(0, min(y1, h-1))
            x2 = max(0, min(x2, w))
            y2 = max(0, min(y2, h))
            
            if x2 <= x1 or y2 <= y1:
                return np.zeros((32, 128), dtype=np.uint8) # Return empty if invalid
                
            cropped = image[y1:y2, x1:x2]
            return cropped
        except Exception as e:
            logging.error(f"Error cropping region: {e}")
            return image
        
    def read(self, input_path: str) -> dict:
        path = Path(input_path)
        
        if path.suffix.lower() == '.pdf':
            return self.pdf_processor.process(str(input_path))
        elif path.suffix.lower() in ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']:
            return self.read_image(str(input_path))
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}")
    
    def read_image(self, image_path: str) -> dict:
        try:
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Could not read image at {image_path}")

            processed = self.preprocessor.process(image)
            text = self.backend.recognize(processed)
            cleaned = self.postprocessor.process(text)

            return {
                "text": cleaned,
                "regions": [{"text": cleaned, "bbox": [0, 0, image.shape[1], image.shape[0]], "confidence": 1.0}],
                "image_path": image_path
            }
        except Exception as e:
            logging.error(f"Error in read_image: {e}")
            return {"text": "", "regions": [], "error": str(e)}

    @torch.no_grad()
    def recognize(self, image):
        try:
            processed = self.preprocessor.process(image)
            text = self.backend.recognize(processed)
            cleaned = self.postprocessor.process(text)
            return {"text": cleaned, "confidence": 1.0}
        except Exception as e:
            logging.error(f"Error in recognize: {e}")
            return {"text": "", "confidence": 0.0}
