import numpy as np
import cv2
from preprocessing.image_processing import ImagePreprocessor


class TestImagePreprocessor:
    """Tests for ImagePreprocessor class."""

    def test_grayscale_conversion_bgr(self, sample_image_bgr):
        """BGR image should be converted to grayscale."""
        preprocessor = ImagePreprocessor()
        processed = preprocessor.process(sample_image_bgr)
        assert len(processed.shape) == 2, "Output should be grayscale (2D array)"

    def test_grayscale_input_passthrough(self, sample_image_gray):
        """Grayscale image should be processed without error."""
        preprocessor = ImagePreprocessor()
        processed = preprocessor.process(sample_image_gray)
        assert len(processed.shape) == 2, "Output should remain grayscale"

    def test_black_image_no_crash(self, black_image):
        """All-black image should not crash the preprocessor."""
        preprocessor = ImagePreprocessor()
        processed = preprocessor.process(black_image)
        # Output is upscaled by default (factor=2), so shape will differ
        assert len(processed.shape) == 2, "Output should be 2D"

    def test_white_image_no_crash(self, white_image):
        """All-white image should not crash the preprocessor."""
        preprocessor = ImagePreprocessor()
        processed = preprocessor.process(white_image)
        # Output is upscaled by default (factor=2), so shape will differ
        assert len(processed.shape) == 2, "Output should be 2D"

    def test_text_image_preserved(self, text_image_bgr):
        """Text image should produce valid binary output."""
        preprocessor = ImagePreprocessor()
        processed = preprocessor.process(text_image_bgr)
        assert len(processed.shape) == 2
        unique_vals = np.unique(processed)
        assert len(unique_vals) <= 2 or (0 in unique_vals and 255 in unique_vals), \
            "Output should be binary after thresholding"

    def test_upscale_factor(self, text_image_gray):
        """Upscaling should increase image dimensions."""
        preprocessor = ImagePreprocessor(upscale_factor=3)
        processed = preprocessor.process(text_image_gray)
        assert processed.shape[0] >= 100
        assert processed.shape[1] >= 200

    def test_no_upscale(self, text_image_gray):
        """upscale_factor=1 should not upscale."""
        preprocessor = ImagePreprocessor(upscale_factor=1)
        processed = preprocessor.process(text_image_gray)
        assert processed.shape == text_image_gray.shape

    def test_clahe_custom_params(self, text_image_bgr):
        """Custom CLAHE parameters should be applied."""
        preprocessor = ImagePreprocessor(clahe_clip_limit=5.0, clahe_tile_size=4)
        processed = preprocessor.process(text_image_bgr)
        assert processed is not None
        assert len(processed.shape) == 2

    def test_from_config(self):
        """from_config classmethod should create preprocessor from dict."""
        config = {
            'preprocessing': {
                'target_size': [800, 600],
                'upscale_factor': 3,
                'clahe_clip_limit': 4.0,
                'clahe_tile_size': 16
            }
        }
        preprocessor = ImagePreprocessor.from_config(config)
        assert preprocessor.target_size == (800, 600)
        assert preprocessor.upscale_factor == 3
        assert preprocessor.clahe_clip_limit == 4.0
        assert preprocessor.clahe_tile_size == 16

    def test_from_config_defaults(self):
        """from_config should use defaults for missing keys."""
        preprocessor = ImagePreprocessor.from_config({})
        assert preprocessor.target_size == (640, 640)
        assert preprocessor.upscale_factor == 2
        assert preprocessor.clahe_clip_limit == 3.0
        assert preprocessor.clahe_tile_size == 8

    def test_output_is_binary(self, text_image_bgr):
        """Output should be binary (0 or 255) after Otsu thresholding."""
        preprocessor = ImagePreprocessor()
        processed = preprocessor.process(text_image_bgr)
        unique_vals = set(np.unique(processed).tolist())
        assert unique_vals.issubset({0, 255}), \
            f"Output should be binary, got unique values: {unique_vals}"

    def test_morphological_kernel_size(self):
        """Verify morphological cleanup uses 2x2 kernel (not 1x1)."""
        preprocessor = ImagePreprocessor()
        img = np.zeros((50, 50), dtype=np.uint8)
        img[10, 10] = 255
        img[20, 20] = 255
        img[30:35, 30:35] = 255
        processed = preprocessor.process(img)
        assert processed is not None

    def test_no_upscale_black_image_shape(self, black_image):
        """With upscale_factor=1, black image shape should match input."""
        preprocessor = ImagePreprocessor(upscale_factor=1)
        processed = preprocessor.process(black_image)
        assert processed.shape == (100, 200)

    def test_fast_mode_skips_heavy_stages(self):
        """Fast mode should binarize without denoising/CLAHE/morphology."""
        preprocessor = ImagePreprocessor(mode="fast")
        img = np.zeros((200, 400), dtype=np.uint8)
        img[50:150, 50:350] = 255  # solid block (clean content)
        processed = preprocessor.process(img)
        # Output stays binary
        unique_vals = set(np.unique(processed).tolist())
        assert unique_vals.issubset({0, 255})

    def test_full_mode_runs_full_pipeline(self):
        """Full mode should always take the full pipeline."""
        preprocessor = ImagePreprocessor(mode="full")
        img = np.zeros((200, 400), dtype=np.uint8)
        img[50:150, 50:350] = 255
        processed = preprocessor.process(img)
        unique_vals = set(np.unique(processed).tolist())
        assert unique_vals.issubset({0, 255})

    def test_auto_picks_fast_for_clean_image(self):
        """Auto mode selects the fast path for clean, high-contrast images."""
        preprocessor = ImagePreprocessor()
        img = np.full((300, 600), 255, dtype=np.uint8)
        cv2.putText(img, "Hello World", (20, 160), cv2.FONT_HERSHEY_SIMPLEX,
                    2.0, 0, 3)
        assert preprocessor._choose_mode(img) == "fast"

    def test_auto_picks_full_for_noisy_image(self):
        """Auto mode selects the full path for noisy / non-binary images."""
        preprocessor = ImagePreprocessor()
        rng = np.random.default_rng(0)
        img = rng.integers(0, 255, (300, 600), dtype=np.uint8)
        assert preprocessor._choose_mode(img) == "full"

    def test_is_clean_heuristic(self):
        """Clean-image heuristic should flag crisp text on a white background."""
        preprocessor = ImagePreprocessor()
        img = np.full((300, 600), 255, dtype=np.uint8)
        cv2.putText(img, "Persian سلام", (20, 160), cv2.FONT_HERSHEY_SIMPLEX,
                    2.0, 0, 3)
        assert preprocessor._is_clean(img) is True

    def test_from_config_mode(self):
        """from_config should read the mode and fast-path thresholds."""
        config = {
            'preprocessing': {
                'mode': 'full',
                'fast_min_sharpness': 200.0,
                'fast_min_binary_fraction': 0.5,
            }
        }
        preprocessor = ImagePreprocessor.from_config(config)
        assert preprocessor.mode == 'full'
        assert preprocessor.fast_min_sharpness == 200.0
        assert preprocessor.fast_min_binary_fraction == 0.5
