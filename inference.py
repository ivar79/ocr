import torch
import cv2
import yaml
import logging
import numpy as np
import os
import shutil
import pytesseract
from pathlib import Path

from preprocessing.image_processing import ImagePreprocessor
from preprocessing.pdf_processing import PDFProcessor
from backend.ocr_backend import OCRBackendFactory

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# Configure Tesseract path - Cross-platform
tesseract_path = shutil.which('tesseract')
if tesseract_path:
    pytesseract.pytesseract.tesseract_cmd = tesseract_path
elif os.name == 'nt':  # Windows fallback
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Set TESSDATA_PREFIX for Windows
if os.name == 'nt':
    os.environ['TESSDATA_PREFIX'] = str(Path.home() / 'AppData' / 'Local' / 'Tesseract-OCR' / 'tessdata')


class OCREngine:
    def __init__(self, config_path: str = "configs/config.yaml"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logging.info(f"Using device: {self.device}")

        self.config = self._load_config(config_path)

        # Preprocessing is handled by backend - no need to duplicate here
        # Keep for PDF pixel-to-numpy conversion if needed
        self.preprocessor = ImagePreprocessor(
            target_size=tuple(self.config.get('preprocessing', {}).get('target_size', [640, 640]))
        )
        
        # PDF processor needs reference to self
        self.pdf_processor = PDFProcessor(self)

        # Backend handles preprocessing and postprocessing internally
        ocr_cfg = self.config.get('ocr', {})
        backend_name = ocr_cfg.get('backend', 'tesseract')
        language = ocr_cfg.get('language', 'fas+eng')
        backend_kwargs = {
            'enable_retry': ocr_cfg.get('enable_retry', True),
            'confidence_threshold': ocr_cfg.get('confidence_threshold', 0.5),
            'max_retries': ocr_cfg.get('max_retries', 2),
        }
        if backend_name.lower() == 'tesseract':
            backend_kwargs['preprocessing_config'] = self.config.get('preprocessing', {})
        else:
            recognition_cfg = self.config.get('recognition', {})
            backend_kwargs['weights'] = (
                ocr_cfg.get('crnn_weights')
                or recognition_cfg.get('weights', 'models/pretrained/crnn_weights.pth')
            )
        self.backend = OCRBackendFactory.create(
            backend=backend_name, language=language, **backend_kwargs)
    
    def _load_config(self, config_path):
        default_config = {
            'ocr': {'backend': 'tesseract', 'language': 'fas+eng'},
            'detection': {'weights': 'models/pretrained/craft_weights.pth'},
            'recognition': {'weights': 'models/pretrained/crnn_weights.pth'},
            'preprocessing': {'target_size': [640, 640], 'clahe_clip_limit': 3.0}
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

    def _crop_region(self, image, box):
        """Crop the image given a bounding box [x1, y1, x2, y2]"""
        try:
            h, w = image.shape[:2]
            x1, y1, x2, y2 = map(int, box)
            
            # Clamp to image dimensions
            x1 = max(0, min(x1, w - 1))
            y1 = max(0, min(y1, h - 1))
            x2 = max(0, min(x2, w))
            y2 = max(0, min(y2, h))
            
            if x2 <= x1 or y2 <= y1:
                return np.zeros((32, 128), dtype=np.uint8)
                
            cropped = image[y1:y2, x1:x2]
            return cropped
        except Exception as e:
            logging.error(f"Error cropping region: {e}")
            return image
        
    def read(self, input_path: str) -> dict:
        path = Path(input_path)
        
        if path.suffix.lower() == '.pdf':
            return self.pdf_processor.process(str(input_path))
        elif path.suffix.lower() in ['.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp']:
            return self.read_image(str(input_path))
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}")
    
    def read_image(self, image_path: str) -> dict:
        try:
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Could not read image at {image_path}")

            # Backend handles preprocessing and postprocessing internally
            # No double processing
            details = self.backend.recognize_with_details(image)
            text = details.get('text', '')
            confidence = details.get('confidence', 1.0)

            return {
                "text": text,
                "confidence": confidence,
                "regions": [{"text": text, "bbox": [0, 0, image.shape[1], image.shape[0]], "confidence": confidence}],
                "image_path": image_path
            }
        except Exception as e:
            logging.error(f"Error in read_image: {e}")
            return {"text": "", "regions": [], "error": str(e)}

    def recognize(self, image):
        """Recognize text from numpy image array"""
        try:
            # Backend handles preprocessing and postprocessing internally
            details = self.backend.recognize_with_details(image)
            return {"text": details.get('text', ''), "confidence": details.get('confidence', 1.0)}
        except Exception as e:
            logging.error(f"Error in recognize: {e}")
            return {"text": "", "confidence": 0.0}
