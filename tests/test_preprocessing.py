import numpy as np
from preprocessing.image_processing import ImagePreprocessor

def test_grayscale_conversion(sample_image):
    preprocessor = ImagePreprocessor()
    processed = preprocessor.process(sample_image)
    assert len(processed.shape) == 2, "Output should be grayscale (2D array)"

def test_black_image(black_image):
    preprocessor = ImagePreprocessor()
    processed = preprocessor.process(black_image)
    assert processed.shape == (100, 200), "Should handle black images without crashing"
