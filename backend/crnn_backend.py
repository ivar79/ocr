"""Deep-learning recognition backend (CRNN).

Layout — finding text lines on the page — is delegated to Tesseract (a fast
single ``image_to_data`` pass, PSM 6). Each detected text line is then
cropped and recognized by the trained CRNN model. This hybrid pipeline
avoids the need for a separate detection model (CRAFT) while giving the
project a real deep-learning recognition path.
"""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np
import pytesseract

from utils.tesseract_setup import configure_tesseract

configure_tesseract()
logger = logging.getLogger(__name__)


class CRNNOCRBackend:
    """CRNN-based recognition with Tesseract line detection."""

    def __init__(self, weights: str = "models/pretrained/crnn_weights.pth",
                 language: str = "fas+eng", device: str = "auto",
                 min_word_conf: int = 30, line_padding: int = 2):
        import torch
        from models.recognition.crnn import CRNN
        from utils.dataset import CharsetManager

        weights_path = Path(weights)
        if not weights_path.exists():
            raise FileNotFoundError(
                f"CRNN weights not found at {weights}. Train with "
                "`python train.py --mode recognition` first.")

        self.charset = CharsetManager()
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # from_checkpoint reads the architecture from the checkpoint file, so
        # training-time choices (hidden_size, cnn_scale) carry over exactly.
        self.model = CRNN.from_checkpoint(str(weights_path), map_location=self.device)
        if self.model.num_classes != self.charset.num_classes:
            logger.warning(
                "Checkpoint charset (%d) differs from current charset (%d); "
                "decoding may be off.", self.model.num_classes, self.charset.num_classes)
        self.model.to(self.device).eval()

        self.language = language
        self.min_word_conf = min_word_conf
        self.line_padding = line_padding

    @staticmethod
    def _load_image(image) -> np.ndarray:
        if isinstance(image, (str, Path)):
            image = cv2.imread(str(image))
        if image is None:
            raise ValueError("Image could not be read.")
        return image

    def _extract_lines(self, gray: np.ndarray) -> list[tuple]:
        """Find text-line bounding boxes using a fast Tesseract layout pass."""
        config = "--oem 3 --psm 6"
        data = pytesseract.image_to_data(
            gray, lang=self.language, config=config,
            output_type=pytesseract.Output.DICT)

        boxes: list[tuple[int, int, int, int]] = []
        n = len(data.get('text', []))
        current_key = None
        current_box = None
        for i in range(n):
            word = (data['text'][i] or '').strip()
            if not word:
                continue
            conf = int(data['conf'][i])
            if conf < self.min_word_conf:
                continue
            x, y, w, h = (int(data['left'][i]), int(data['top'][i]),
                          int(data['width'][i]), int(data['height'][i]))
            key = (data['block_num'][i], data['par_num'][i], data['line_num'][i])
            if key != current_key:
                if current_box is not None:
                    boxes.append(current_box)
                current_key = key
                current_box = [x, y, x + w, y + h]
            else:
                current_box[0] = min(current_box[0], x)
                current_box[1] = min(current_box[1], y)
                current_box[2] = max(current_box[2], x + w)
                current_box[3] = max(current_box[3], y + h)
        if current_box is not None:
            boxes.append(current_box)

        # Fallbacks for single-line / small inputs: PSM 6 often finds no words
        # on a lone text line, so retry with sparse layout and finally treat
        # the whole image as a single line.
        if not boxes:
            h, w = gray.shape[:2]
            boxes = [(0, 0, w, h)]
        return boxes

    @staticmethod
    def _prepare_crop(crop: np.ndarray) -> np.ndarray:
        """Binarize a line crop so text is dark on a white background."""
        _, binary = cv2.threshold(
            crop, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # Otsu keeps polarity; force dark text on white bg for the CRNN.
        if np.mean(binary) < 127:
            binary = 255 - binary
        return binary

    def recognize(self, image) -> str:
        raw = self._load_image(image)
        gray = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY) if len(raw.shape) == 3 else raw
        boxes = self._extract_lines(gray)
        if not boxes:
            return ""

        texts: list[str] = []
        confs: list[float] = []
        pad = self.line_padding
        h_img, w_img = gray.shape[:2]
        for x1, y1, x2, y2 in boxes:
            x1 = max(0, x1 - pad)
            y1 = max(0, y1 - pad)
            x2 = min(w_img, x2 + pad)
            y2 = min(h_img, y2 + pad)
            crop = gray[y1:y2, x1:x2]
            if crop.size == 0 or crop.shape[0] < 8 or crop.shape[1] < 8:
                continue
            prepared = self._prepare_crop(crop)
            text = self.model.recognize(prepared)
            texts.append(text)
            confs.append(float(getattr(self.model, 'last_confidence', 1.0)))
        return "\n".join(texts)

    def recognize_with_details(self, image) -> dict:
        raw = self._load_image(image)
        gray = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY) if len(raw.shape) == 3 else raw
        boxes = self._extract_lines(gray)
        texts: list[str] = []
        confs: list[float] = []
        pad = self.line_padding
        h_img, w_img = gray.shape[:2]
        for x1, y1, x2, y2 in boxes:
            x1 = max(0, x1 - pad)
            y1 = max(0, y1 - pad)
            x2 = min(w_img, x2 + pad)
            y2 = min(h_img, y2 + pad)
            crop = gray[y1:y2, x1:x2]
            if crop.size == 0 or crop.shape[0] < 8 or crop.shape[1] < 8:
                continue
            prepared = self._prepare_crop(crop)
            text = self.model.recognize(prepared)
            texts.append(text)
            confs.append(float(getattr(self.model, 'last_confidence', 1.0)))
        avg_conf = sum(confs) / len(confs) if confs else 0.0
        return {
            'text': "\n".join(texts),
            'confidence': avg_conf,
            'psm_used': None,
            'image_analysis': {'num_lines': len(boxes)},
        }
