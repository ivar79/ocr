import cv2
import numpy as np

class ImagePreprocessor:
    def __init__(self, target_size=(640, 640), upscale_factor=2):
        self.target_size = target_size
        self.upscale_factor = upscale_factor
    
    def process(self, image: np.ndarray) -> np.ndarray:
        # 1. Grayscale conversion
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # 2. Upscaling (for low-res images)
        if self.upscale_factor > 1:
            gray = cv2.resize(gray, None, fx=self.upscale_factor, fy=self.upscale_factor, interpolation=cv2.INTER_CUBIC)

        # 3. Noise Reduction (Bilateral filter preserves edges better than Gaussian)
        denoised = cv2.bilateralFilter(gray, d=9, sigmaColor=75, sigmaSpace=75)
        
        # 4. Contrast Enhancement (CLAHE) - increased for better text contrast
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(denoised)
        
        # 5. Skip aggressive thresholding for Persian text to preserve character connections
        # Use mild thresholding instead
        _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # 6. Deskewing (before morphological operations)
        corrected = self._deskew(binary)
        
        # 7. Mild Morphological Cleanup (very gentle to preserve Persian connections)
        cleaned = self._morphological_cleanup_mild(corrected)
        
        return cleaned
    
    def _morphological_cleanup_mild(self, image: np.ndarray) -> np.ndarray:
        """Very mild morphological cleanup to preserve Persian character connections."""
        # Use very small kernel for Persian text preservation
        kernel = np.ones((1, 1), np.uint8)
        # Only remove single-pixel noise
        cleaned = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
        return cleaned

    def _morphological_cleanup(self, image: np.ndarray) -> np.ndarray:
        kernel = np.ones((2, 2), np.uint8)
        # Remove tiny dots (Noise)
        opening = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
        # Connect broken parts of characters (Closing)
        closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
        return closing

    def _deskew(self, image: np.ndarray) -> np.ndarray:
        coords = np.column_stack(np.where(image > 0))
        if len(coords) == 0:
            return image
            
        angle = cv2.minAreaRect(coords)[-1]
        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle
            
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(
            image, M, (w, h),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE
        )
        return rotated
