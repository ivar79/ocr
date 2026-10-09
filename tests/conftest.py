import pytest
import numpy as np
import cv2
import os
import tempfile
from pathlib import Path


@pytest.fixture
def sample_image_bgr():
    """Random BGR image for testing."""
    return np.random.randint(0, 255, (100, 200, 3), dtype=np.uint8)


@pytest.fixture
def sample_image_gray():
    """Random grayscale image for testing."""
    return np.random.randint(0, 255, (100, 200), dtype=np.uint8)


@pytest.fixture
def black_image():
    """All-black image."""
    return np.zeros((100, 200, 3), dtype=np.uint8)


@pytest.fixture
def white_image():
    """All-white image."""
    return np.ones((100, 200, 3), dtype=np.uint8) * 255


@pytest.fixture
def text_image_bgr():
    """Image with white text on black background (BGR)."""
    img = np.zeros((100, 300, 3), dtype=np.uint8)
    cv2.putText(img, "Hello World", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2)
    return img


@pytest.fixture
def text_image_gray():
    """Image with white text on black background (grayscale)."""
    img = np.zeros((100, 300), dtype=np.uint8)
    cv2.putText(img, "Hello World", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.5, 255, 2)
    return img


@pytest.fixture
def sample_image_path(tmp_path, text_image_bgr):
    """Save a test image and return its path."""
    img_path = tmp_path / "test_image.png"
    cv2.imwrite(str(img_path), text_image_bgr)
    return str(img_path)


@pytest.fixture
def sample_config():
    """Sample config dictionary."""
    return {
        'ocr': {'backend': 'tesseract', 'language': 'fas+eng'},
        'preprocessing': {'target_size': [640, 640], 'clahe_clip_limit': 3.0}
    }


@pytest.fixture
def empty_data_dir(tmp_path):
    """Empty data directory for dataset tests."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "images").mkdir()
    return data_dir
