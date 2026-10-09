from __future__ import annotations

import logging
import os
import urllib.request
from pathlib import Path

import cv2
import numpy as np
import pytesseract

from preprocessing.image_processing import ImagePreprocessor
from postprocessing.text_formatter import TextPostProcessor
from utils.tesseract_setup import configure_tesseract

configure_tesseract()
logger = logging.getLogger(__name__)


class TesseractOCRBackend:
    """Tesseract OCR Backend with adaptive configuration.

    Features:
    - Adaptive PSM selection based on image analysis
    - Confidence-based retry with different settings (keeps the best result)
    - Single-pass OCR: one ``image_to_data`` call per attempt provides both
      text and confidence (no duplicate Tesseract runs)
    - Automatic language data management
    """

    _fas_checked: bool = False

    PSM_MODES = {
        'auto': 3,
        'single_block': 6,
        'single_line': 7,
        'single_word': 8,
        'sparse': 11,
    }

    # Alternative PSM values tried when confidence is low
    RETRY_PSMS = [3, 6, 11]

    def __init__(self, language: str = "fas+eng", enable_retry: bool = True,
                 confidence_threshold: float = 0.5, max_retries: int = 2,
                 preprocessing_config: dict | None = None):
        self.language = language
        self.enable_retry = enable_retry
        self.confidence_threshold = confidence_threshold
        self.max_retries = max_retries
        self._ensure_language_data(language)
        if preprocessing_config:
            self.preprocessor = ImagePreprocessor.from_config(
                {'preprocessing': preprocessing_config})
        else:
            self.preprocessor = ImagePreprocessor()
        self.postprocessor = TextPostProcessor()

    @staticmethod
    def _candidate_tessdata_dirs() -> list[Path]:
        dirs: list[Path] = []
        env_prefix = os.environ.get("TESSDATA_PREFIX")
        if env_prefix:
            dirs.append(Path(env_prefix))
        dirs.extend([
            Path(r"C:\Program Files\Tesseract-OCR\tessdata"),
            Path.home() / "AppData" / "Local" / "Tesseract-OCR" / "tessdata",
            Path.cwd() / "tessdata",
            Path("/usr/share/tesseract-ocr/5/tessdata"),
            Path("/usr/share/tesseract-ocr/4.00/tessdata"),
        ])
        unique: list[Path] = []
        seen: set[str] = set()
        for directory in dirs:
            resolved = directory.resolve(strict=False)
            key = str(resolved)
            if key not in seen:
                unique.append(resolved)
                seen.add(key)
        return unique

    @classmethod
    def _ensure_language_data(cls, language: str) -> None:
        if not language:
            return
        requested = [part.strip() for part in language.split("+") if part.strip()]
        for lang in requested:
            if lang == "eng":
                continue
            if lang == "fas":
                if cls._fas_checked:
                    continue
                if any((directory / "fas.traineddata").exists() for directory in cls._candidate_tessdata_dirs()):
                    cls._fas_checked = True
                    continue
                local_dir = Path.home() / "AppData" / "Local" / "Tesseract-OCR" / "tessdata"
                local_dir.mkdir(parents=True, exist_ok=True)
                target = local_dir / "fas.traineddata"
                url = "https://github.com/tesseract-ocr/tessdata_fast/raw/main/fas.traineddata"
                try:
                    urllib.request.urlretrieve(url, str(target))
                except Exception:
                    url = "https://github.com/tesseract-ocr/tessdata/raw/main/fas.traineddata"
                    try:
                        urllib.request.urlretrieve(url, str(target))
                    except Exception:
                        raise RuntimeError("Persian Tesseract language data is missing.")
                os.environ["TESSDATA_PREFIX"] = str(local_dir)
                cls._fas_checked = True

    def _load_image(self, image) -> np.ndarray:
        if isinstance(image, (str, Path)):
            image = cv2.imread(str(image))
        if image is None:
            raise ValueError("Image could not be read.")
        return image

    def _analyze_image(self, image: np.ndarray) -> dict:
        h, w = image.shape[:2]
        mean_val = np.mean(image)
        std_val = np.std(image)
        if h < 50 or w < 100:
            psm = self.PSM_MODES['single_word']
        elif h < 100:
            psm = self.PSM_MODES['single_line']
        elif std_val < 30:
            psm = self.PSM_MODES['sparse']
        else:
            psm = self.PSM_MODES['single_block']
        return {
            'width': w,
            'height': h,
            'mean_brightness': float(mean_val),
            'contrast': float(std_val),
            'recommended_psm': psm,
        }

    def _get_config(self, psm: int = None) -> str:
        if psm is None:
            psm = self.PSM_MODES['single_block']
        return f"--oem 3 --psm {psm}"

    @staticmethod
    def _rebuild_text(data: dict) -> str:
        """Reconstruct multi-line text from an image_to_data dict.

        Words belonging to the same Tesseract line (block/par/line) are
        joined with spaces; lines are joined with newlines.
        """
        n = len(data.get('text', []))
        lines: list[str] = []
        current_words: list[str] = []
        prev_line_key = None
        for i in range(n):
            word = data['text'][i]
            if not word or not word.strip():
                continue
            line_key = (data['block_num'][i], data['par_num'][i], data['line_num'][i])
            if prev_line_key is not None and line_key != prev_line_key:
                lines.append(' '.join(current_words))
                current_words = []
            prev_line_key = line_key
            current_words.append(word.strip())
        if current_words:
            lines.append(' '.join(current_words))
        return '\n'.join(lines)

    def _ocr_with_confidence(self, image: np.ndarray, psm: int) -> tuple:
        """Run OCR once and return (text, confidence) in a single Tesseract pass.

        ``image_to_data`` returns both the recognized words and per-word
        confidence, so there is no need for a second ``image_to_string`` call.
        """
        config = self._get_config(psm)
        try:
            data = pytesseract.image_to_data(
                image, lang=self.language, config=config,
                output_type=pytesseract.Output.DICT)
            text = self._rebuild_text(data)
            # Fall back to image_to_string if the TSV path produced nothing
            if not text.strip():
                text = pytesseract.image_to_string(image, lang=self.language, config=config)
            confidences = [c for c in data['conf'] if c > 0]
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
            return text, avg_confidence / 100.0
        except Exception:
            text = pytesseract.image_to_string(image, lang=self.language, config=config)
            return text, 0.5

    def _recognize_best(self, image: np.ndarray, primary_psm: int) -> tuple:
        """Run OCR, retrying with alternative PSMs when confidence is low.

        Returns (text, confidence, psm_used). The best result across attempts
        (highest confidence with non-empty text) is kept.
        """
        text, confidence = self._ocr_with_confidence(image, primary_psm)
        best_text, best_conf, best_psm = text, confidence, primary_psm

        if self.enable_retry and confidence < self.confidence_threshold:
            alternatives = [p for p in self.RETRY_PSMS if p != primary_psm][:self.max_retries]
            for alt_psm in alternatives:
                alt_text, alt_confidence = self._ocr_with_confidence(image, alt_psm)
                if len(alt_text.strip()) > 0 and alt_confidence > best_conf:
                    best_text, best_conf, best_psm = alt_text, alt_confidence, alt_psm
                if best_conf >= self.confidence_threshold:
                    break
        return best_text, best_conf, best_psm

    def recognize(self, image) -> str:
        raw_image = self._load_image(image)
        processed = self.preprocessor.process(raw_image)
        analysis = self._analyze_image(processed)
        text, confidence, _ = self._recognize_best(processed, analysis['recommended_psm'])
        final_text = self.postprocessor.process(text)
        return final_text

    def recognize_with_details(self, image) -> dict:
        raw_image = self._load_image(image)
        processed = self.preprocessor.process(raw_image)
        analysis = self._analyze_image(processed)
        text, confidence, psm = self._recognize_best(processed, analysis['recommended_psm'])
        final_text = self.postprocessor.process(text)
        return {
            'text': final_text,
            'confidence': confidence,
            'psm_used': psm,
            'image_analysis': analysis,
        }


class OCRBackendFactory:
    @staticmethod
    def create(backend: str = "tesseract", language: str = "fas+eng", **kwargs):
        name = backend.lower()
        if name == "tesseract":
            return TesseractOCRBackend(language=language, **kwargs)
        if name == "crnn":
            from backend.crnn_backend import CRNNOCRBackend
            try:
                return CRNNOCRBackend(language=language, **kwargs)
            except Exception as e:
                logger.warning(
                    "CRNN backend unavailable (%s); falling back to Tesseract.", e)
                return TesseractOCRBackend(language=language, **kwargs)
        raise ValueError(f"Unsupported OCR backend: {backend}")
