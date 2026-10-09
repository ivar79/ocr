import cv2
import numpy as np


class ImagePreprocessor:
    """Image preprocessing for OCR - optimized for Persian text.

    Features:
    - Adaptive processing: fast path for clean images, full pipeline for noisy ones
    - Fast bilateral filter with fallback
    - CLAHE with configurable parameters
    - Smart deskewing
    """

    def __init__(self, target_size=(640, 640), upscale_factor=2,
                 clahe_clip_limit=3.0, clahe_tile_size=8,
                 denoise_strength=75, enable_fast_denoise=True,
                 mode="auto", fast_min_sharpness=150.0,
                 fast_min_binary_fraction=0.4, max_deskew_area=2_000_000):
        """Initialize preprocessor.

        Args:
            target_size: Target image size (reserved for future)
            upscale_factor: Upscale factor for low-res images
            clahe_clip_limit: CLAHE clip limit for contrast enhancement
            clahe_tile_size: CLAHE tile grid size
            denoise_strength: Bilateral filter sigmaColor/sigmaSpace
            enable_fast_denoise: Use fast denoising for large images
            mode: 'auto' (pick path from image analysis), 'fast', or 'full'
            fast_min_sharpness: Min Laplacian variance for the fast path
            fast_min_binary_fraction: Min fraction of near-binary pixels for the fast path
            max_deskew_area: Skip deskewing for images larger than this (pixels)
        """
        self.target_size = target_size
        self.upscale_factor = upscale_factor
        self.clahe_clip_limit = clahe_clip_limit
        self.clahe_tile_size = clahe_tile_size
        self.denoise_strength = denoise_strength
        self.enable_fast_denoise = enable_fast_denoise
        self.mode = mode
        self.fast_min_sharpness = fast_min_sharpness
        self.fast_min_binary_fraction = fast_min_binary_fraction
        self.max_deskew_area = max_deskew_area

    @classmethod
    def from_config(cls, config: dict):
        """Create instance from config dictionary."""
        preprocessing = config.get('preprocessing', {})
        return cls(
            target_size=tuple(preprocessing.get('target_size', [640, 640])),
            upscale_factor=preprocessing.get('upscale_factor', 2),
            clahe_clip_limit=preprocessing.get('clahe_clip_limit', 3.0),
            clahe_tile_size=preprocessing.get('clahe_tile_size', 8),
            denoise_strength=preprocessing.get('denoise_strength', 75),
            enable_fast_denoise=preprocessing.get('enable_fast_denoise', True),
            mode=preprocessing.get('mode', 'auto'),
            fast_min_sharpness=preprocessing.get('fast_min_sharpness', 150.0),
            fast_min_binary_fraction=preprocessing.get('fast_min_binary_fraction', 0.4),
            max_deskew_area=preprocessing.get('max_deskew_area', 2_000_000),
        )

    def process(self, image: np.ndarray) -> np.ndarray:
        """Process image for OCR with adaptive preprocessing.

        Clean, high-quality images (screenshots, crisp scans) take the fast
        path: grayscale -> (optional upscale) -> Otsu binarization. This is
        both much faster and preserves the antialiased strokes that aggressive
        denoising/CLAHE/morphology destroy on Persian text.

        Noisy, low-contrast, or small images take the full pipeline:
        1. Grayscale conversion
        2. Smart upscaling (based on image size)
        3. Adaptive noise reduction
        4. Contrast enhancement (CLAHE)
        5. Adaptive thresholding
        6. Deskewing
        7. Morphological cleanup
        """
        # 1. Grayscale conversion
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        if self._choose_mode(gray) == 'fast':
            return self._process_fast(gray)
        return self._process_full(gray)

    def _choose_mode(self, gray: np.ndarray) -> str:
        """Decide between the fast and full pipelines."""
        if self.mode == 'fast':
            return 'fast'
        if self.mode == 'full':
            return 'full'
        if self._is_clean(gray):
            return 'fast'
        return 'full'

    def _is_clean(self, gray: np.ndarray) -> bool:
        """Heuristic: is this a clean, high-quality image (screenshot/crisp scan)?"""
        h, w = gray.shape[:2]
        if h < 30 or w < 30:
            return False
        sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        binary_fraction = float(np.mean((gray < 30) | (gray > 225)))
        return (sharpness >= self.fast_min_sharpness
                and binary_fraction >= self.fast_min_binary_fraction)

    def _process_fast(self, gray: np.ndarray) -> np.ndarray:
        """Minimal processing for clean images: upscale + Otsu only."""
        h, w = gray.shape[:2]
        if self.upscale_factor > 1 and (h < 300 or w < 300):
            gray = self._smart_upscale(gray)
        # Otsu keeps the output binary (antialiased screenshots binarize
        # cleanly) without denoising/CLAHE/morphology.
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return binary

    def _process_full(self, gray: np.ndarray) -> np.ndarray:
        """Full pipeline for noisy / low-contrast / small images."""
        h, w = gray.shape[:2]

        # 2. Smart upscaling - only if image is small
        if self.upscale_factor > 1 and (h < 300 or w < 300):
            gray = self._smart_upscale(gray)

        # 3. Adaptive noise reduction
        denoised = self._adaptive_denoise(gray)

        # 4. Contrast Enhancement (CLAHE)
        enhanced = self._apply_clahe(denoised)

        # 5. Adaptive thresholding
        binary = self._adaptive_threshold(enhanced)

        # 6. Deskewing (conditional)
        corrected = self._deskew(binary)

        # 7. Morphological cleanup
        cleaned = self._morphological_cleanup(corrected)

        return cleaned

    def _smart_upscale(self, gray: np.ndarray) -> np.ndarray:
        """Intelligently upscale based on image content."""
        h, w = gray.shape[:2]

        # Calculate target size (keep aspect ratio)
        target_h = max(h * self.upscale_factor, 300)
        target_w = max(w * self.upscale_factor, 300)

        # INTER_LINEAR is nearly as good as INTER_CUBIC for OCR and much faster
        upscaled = cv2.resize(gray, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        return upscaled

    def _adaptive_denoise(self, gray: np.ndarray) -> np.ndarray:
        """Adaptive noise reduction based on image size."""
        h, w = gray.shape[:2]
        pixel_count = h * w

        # For large images, use faster denoising
        if self.enable_fast_denoise and pixel_count > 500000:
            # Fast approximation: Gaussian blur + sharpening
            blurred = cv2.GaussianBlur(gray, (3, 3), 0)
            denoised = cv2.addWeighted(gray, 1.5, blurred, -0.5, 0)
            return denoised

        # For smaller images, use bilateral filter (better quality)
        d = min(9, max(3, int(min(h, w) / 100)))
        denoised = cv2.bilateralFilter(
            gray, d=d,
            sigmaColor=self.denoise_strength,
            sigmaSpace=self.denoise_strength
        )
        return denoised

    def _apply_clahe(self, gray: np.ndarray) -> np.ndarray:
        """Apply CLAHE with configured parameters."""
        clahe = cv2.createCLAHE(
            clipLimit=self.clahe_clip_limit,
            tileGridSize=(self.clahe_tile_size, self.clahe_tile_size)
        )
        return clahe.apply(gray)

    def _adaptive_threshold(self, gray: np.ndarray) -> np.ndarray:
        """Choose best thresholding method based on image content."""
        # Calculate image statistics
        mean_val = np.mean(gray)
        std_val = np.std(gray)

        # If image has good contrast, use Otsu
        if std_val > 40:
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        else:
            # Low contrast: use adaptive threshold
            binary = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY, 11, 2
            )

        return binary

    def _deskew(self, image: np.ndarray) -> np.ndarray:
        """Correct image skew (rotation).

        Only applied when the angle is confident and the image is small
        enough that the whole-foreground bounding box is meaningful
        (single blocks / crops). Large multi-line pages are left untouched.
        """
        h, w = image.shape[:2]
        if h * w > self.max_deskew_area:
            return image

        coords = np.column_stack(np.where(image > 0))
        if len(coords) < 100:  # Not enough points
            return image

        try:
            angle = cv2.minAreaRect(coords)[-1]
            if angle < -45:
                angle = -(90 + angle)
            else:
                angle = -angle

            # Skip if angle is very small
            if abs(angle) < 0.5:
                return image

            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            rotated = cv2.warpAffine(
                image, M, (w, h),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REPLICATE
            )
            return rotated
        except Exception:
            return image

    def _morphological_cleanup(self, image: np.ndarray) -> np.ndarray:
        """Mild morphological cleanup to preserve Persian connections."""
        kernel = np.ones((2, 2), np.uint8)
        cleaned = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
        return cleaned

    def analyze_image(self, image: np.ndarray) -> dict:
        """Analyze image quality and recommend preprocessing settings.

        Returns:
            Dictionary with quality metrics and recommendations
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        h, w = gray.shape[:2]
        mean_val = np.mean(gray)
        std_val = np.std(gray)
        blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()

        return {
            'width': w,
            'height': h,
            'mean_brightness': float(mean_val),
            'contrast': float(std_val),
            'sharpness': float(blur_score),
            'is_low_contrast': std_val < 40,
            'is_blurry': blur_score < 100,
            'is_small': h < 300 or w < 300,
            'recommended_upscale': 2 if (h < 300 or w < 300) else 1,
            'recommended_mode': 'fast' if self._is_clean(gray) else 'full',
        }
