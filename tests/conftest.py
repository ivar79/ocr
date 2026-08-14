import pytest
import numpy as np

@pytest.fixture
def sample_image():
    # تصویر رندوم BGR
    return np.random.randint(0, 255, (100, 200, 3), dtype=np.uint8)

@pytest.fixture
def black_image():
    return np.zeros((100, 200, 3), dtype=np.uint8)
