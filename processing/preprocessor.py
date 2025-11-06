import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
from typing import List, Dict, Any

class DocumentPreprocessor:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
    
    def enhance_image_quality(self, image: np.ndarray) -> np.ndarray:
        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        enhancer = ImageEnhance.Contrast(pil_image)
        pil_image = enhancer.enhance(1.5)
        
        enhancer = ImageEnhance.Sharpness(pil_image)
        pil_image = enhancer.enhance(2.0)
        
        if self.config.get('denoise', True):
            pil_image = pil_image.filter(ImageFilter.MedianFilter(size=3))
        
        return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    
    def deskew_image(self, image: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.bitwise_not(gray)
        
        coords = np.column_stack(np.where(gray > 0))
        angle = cv2.minAreaRect(coords)[-1]
        
        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle
        
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, 
                               borderMode=cv2.BORDER_REPLICATE)
        
        return rotated
    
    def remove_noise(self, image: np.ndarray) -> np.ndarray:
        denoised = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
        return denoised
    
    def normalize_size(self, image: np.ndarray, target_width: int = 1200) -> np.ndarray:
        height, width = image.shape[:2]
        
        if width > target_width:
            scale_factor = target_width / width
            new_height = int(height * scale_factor)
            resized = cv2.resize(image, (target_width, new_height), 
                               interpolation=cv2.INTER_AREA)
            return resized
        
        return image
    
    def preprocess_document(self, image_path: str) -> Dict[str, Any]:
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image from {image_path}")
        
        original_shape = image.shape
        
        enhanced = self.enhance_image_quality(image)
        deskewed = self.deskew_image(enhanced)
        denoised = self.remove_noise(deskewed)
        normalized = self.normalize_size(denoised)
        
        return {
            'processed_image': normalized,
            'original_shape': original_shape,
            'processing_steps': ['enhancement', 'deskewing', 'denoising', 'normalization']
        }