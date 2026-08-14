from __future__ import annotations

import os
import urllib.request
from pathlib import Path

import cv2
import numpy as np
import pytesseract

from preprocessing.image_processing import ImagePreprocessor
from postprocessing.text_formatter import TextPostProcessor


class TesseractOCRBackend:
    def __init__(self, language: str = "fas+eng"):
        self.language = language
        self._ensure_language_data(language)
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
                if any((directory / "fas.traineddata").exists() for directory in cls._candidate_tessdata_dirs()):
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
                        raise RuntimeError(
                            "Persian Tesseract language data is missing and could not be downloaded. "
                            "Please install fas.traineddata into a writable tessdata directory."
                        )

                os.environ["TESSDATA_PREFIX"] = str(local_dir)

    def _load_image(self, image) -> np.ndarray:
        if isinstance(image, (str, Path)):
            image = cv2.imread(str(image))
        if image is None:
            raise ValueError("Image could not be read.")
        return image

    def recognize(self, image) -> str:
        self._ensure_language_data(self.language)
        
        # 1. Load the image
        raw_image = self._load_image(image)
        
        # 2. Advanced Preprocessing
        processed = self.preprocessor.process(raw_image)
        
        # 3. Language Detection
        detected_lang = self._detect_language(processed)
        config = self._get_optimal_config(detected_lang)
        
        # 4. OCR Execution
        text = pytesseract.image_to_string(processed, lang=detected_lang, config=config)
        
        # 5. Advanced Postprocessing
        final_text = self.postprocessor.process(text)
        
        return final_text

    @staticmethod
    def _detect_language(image: np.ndarray) -> str:
        """Detect if image contains Persian or English text - default to Persian for mixed content."""
        # Default to Persian (fas+eng) for better results on mixed content
        # Persian script detection is more reliable with both languages enabled
        return 'fas+eng'

    @staticmethod
    def _get_optimal_config(language: str) -> str:
        """Get optimized Tesseract config for specific language."""
        base_oem = "--oem 3"
        
        if 'fas' in language:
            # PSM 6: Assume a single uniform block of text (better for Persian)
            psm = "6"
            config = "{} --psm {}".format(base_oem, psm)
        else:
            psm = "6"
            config = "{} --psm {}".format(base_oem, psm)
        
        return config


class OCRBackendFactory:
    @staticmethod
    def create(backend: str = "tesseract", language: str = "fas+eng"):
        if backend.lower() == "tesseract":
            return TesseractOCRBackend(language=language)
        raise ValueError(f"Unsupported OCR backend: {backend}")
