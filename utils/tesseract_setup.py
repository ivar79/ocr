"""Shared Tesseract binary / language-data path configuration.

Backends import this so they work standalone (not only after ``inference``
has been imported).
"""

import os
import shutil
from pathlib import Path

import pytesseract

_configured = False


def configure_tesseract() -> None:
    """Locate the tesseract binary and tessdata directory (cross-platform)."""
    global _configured
    if _configured:
        return

    tesseract_bin = shutil.which('tesseract')
    if tesseract_bin:
        pytesseract.pytesseract.tesseract_cmd = tesseract_bin
    elif os.name == 'nt':
        pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

    if os.name == 'nt':
        os.environ.setdefault(
            'TESSDATA_PREFIX',
            str(Path.home() / 'AppData' / 'Local' / 'Tesseract-OCR' / 'tessdata'))

    _configured = True
