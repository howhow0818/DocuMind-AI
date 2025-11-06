import cv2
import numpy as np
import pytesseract
from PIL import Image
import torch
import torch.nn as nn
from typing import Dict, List, Any, Tuple

class EnhancedOCREngine:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.language = config.get('language', 'eng')
        self.ocr_engine = config.get('ocr_engine', 'tesseract')
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        denoised = cv2.fastNlMeansDenoising(gray)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(denoised)
        
        thresh = cv2.adaptiveThreshold(enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                     cv2.THRESH_BINARY, 11, 2)
        
        kernel = np.ones((1,1), np.uint8)
        processed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        processed = cv2.morphologyEx(processed, cv2.MORPH_OPEN, kernel)
        
        return processed
    
    def detect_text_regions(self, image: np.ndarray) -> List[Dict[str, Any]]:
        processed = self.preprocess_image(image)
        
        d = pytesseract.image_to_data(processed, output_type=pytesseract.Output.DICT, 
                                    config='--psm 6')
        
        regions = []
        n_boxes = len(d['level'])
        for i in range(n_boxes):
            if int(d['conf'][i]) > 30:
                x, y, w, h = d['left'][i], d['top'][i], d['width'][i], d['height'][i]
                text = d['text'][i].strip()
                if text:
                    regions.append({
                        'bbox': [x, y, x+w, y+h],
                        'text': text,
                        'confidence': float(d['conf'][i]) / 100.0
                    })
        
        return regions
    
    def extract_text_with_orientation(self, image: np.ndarray) -> Dict[str, Any]:
        results = {}
        
        for angle in [0, 90, 180, 270]:
            rotated = self.rotate_image(image, angle)
            processed = self.preprocess_image(rotated)
            
            custom_config = r'--oem 3 --psm 6'
            text_data = pytesseract.image_to_data(processed, config=custom_config, 
                                                output_type=pytesseract.Output.DICT)
            
            total_conf = 0
            valid_chars = 0
            for i, conf in enumerate(text_data['conf']):
                if int(conf) > 0:
                    total_conf += int(conf)
                    valid_chars += 1
            
            avg_conf = total_conf / max(valid_chars, 1)
            
            if avg_conf > results.get('confidence', 0):
                results = {
                    'angle': angle,
                    'text': ' '.join([t for t in text_data['text'] if t.strip()]),
                    'confidence': avg_conf / 100.0,
                    'regions': self.detect_text_regions(rotated)
                }
        
        return results
    
    def rotate_image(self, image: np.ndarray, angle: float) -> np.ndarray:
        if angle == 0:
            return image
        
        height, width = image.shape[:2]
        center = (width // 2, height // 2)
        
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(image, rotation_matrix, (width, height), 
                               flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        
        return rotated
    
    def process_document(self, image_path: str) -> Dict[str, Any]:
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image from {image_path}")
        
        orientation_result = self.extract_text_with_orientation(image)
        full_text = self.extract_full_text(image)
        
        return {
            'orientation': orientation_result['angle'],
            'confidence': orientation_result['confidence'],
            'full_text': full_text,
            'text_regions': orientation_result['regions'],
            'image_shape': image.shape
        }
    
    def extract_full_text(self, image: np.ndarray) -> str:
        processed = self.preprocess_image(image)
        
        custom_config = r'--oem 3 --psm 6'
        text = pytesseract.image_to_string(processed, config=custom_config)
        
        return text.strip()