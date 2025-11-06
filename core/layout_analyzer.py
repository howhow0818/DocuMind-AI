import cv2
import numpy as np
from sklearn.cluster import DBSCAN
from typing import Dict, List, Any, Tuple
import torch
import torch.nn as nn

class LayoutAnalyzer:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def detect_contours(self, image: np.ndarray) -> List[np.ndarray]:
        processed = self.preprocess_for_layout(image)
        
        contours, _ = cv2.findContours(processed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        filtered_contours = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 1000:
                filtered_contours.append(contour)
        
        return filtered_contours
    
    def preprocess_for_layout(self, image: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        blurred = cv2.GaussianBlur(gray, (5,5), 0)
        edges = cv2.Canny(blurred, 50, 150)
        
        kernel = np.ones((3,3), np.uint8)
        dilated = cv2.dilate(edges, kernel, iterations=2)
        closed = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel)
        
        return closed
    
    def detect_tables(self, image: np.ndarray) -> List[Dict[str, Any]]:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                     cv2.THRESH_BINARY, 11, 2)
        binary = 255 - binary
        
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 50))
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (50, 1))
        
        vertical_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vertical_kernel, iterations=2)
        horizontal_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
        
        table_structure = cv2.addWeighted(vertical_lines, 0.5, horizontal_lines, 0.5, 0.0)
        table_structure = cv2.threshold(table_structure, 128, 255, cv2.THRESH_BINARY)[1]
        
        contours, _ = cv2.findContours(table_structure, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        tables = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if w > 100 and h > 100:
                tables.append({
                    'bbox': [x, y, x+w, y+h],
                    'area': w * h,
                    'type': 'table'
                })
        
        return tables
    
    def detect_forms(self, image: np.ndarray) -> List[Dict[str, Any]]:
        contours = self.detect_contours(image)
        
        forms = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / h
            
            if 0.5 < aspect_ratio < 2.0 and w > 200 and h > 200:
                forms.append({
                    'bbox': [x, y, x+w, y+h],
                    'area': w * h,
                    'type': 'form_field'
                })
        
        return forms
    
    def analyze_document_layout(self, image: np.ndarray, text_regions: List[Dict]) -> Dict[str, Any]:
        tables = self.detect_tables(image)
        forms = self.detect_forms(image)
        contours = self.detect_contours(image)
        
        regions_by_type = self.classify_regions(text_regions, tables, forms)
        
        reading_order = self.determine_reading_order(regions_by_type['text'])
        
        return {
            'tables': tables,
            'forms': forms,
            'structural_elements': contours,
            'regions_by_type': regions_by_type,
            'reading_order': reading_order,
            'page_structure': self.analyze_page_structure(regions_by_type)
        }
    
    def classify_regions(self, text_regions: List[Dict], tables: List[Dict], forms: List[Dict]) -> Dict[str, List]:
        regions = {
            'text': text_regions,
            'table': tables,
            'form': forms,
            'header': [],
            'footer': [],
            'sidebar': []
        }
        
        image_height = 1000
        for region in text_regions:
            bbox = region['bbox']
            y_center = (bbox[1] + bbox[3]) / 2
            
            if bbox[1] < image_height * 0.15:
                regions['header'].append(region)
            elif bbox[1] > image_height * 0.85:
                regions['footer'].append(region)
            elif bbox[0] < 100 or bbox[2] > 900:
                regions['sidebar'].append(region)
        
        return regions
    
    def determine_reading_order(self, regions: List[Dict]) -> List[Dict]:
        if not regions:
            return []
        
        sorted_regions = sorted(regions, key=lambda x: (x['bbox'][1], x['bbox'][0]))
        
        clusters = []
        current_cluster = [sorted_regions[0]]
        
        for region in sorted_regions[1:]:
            last_region = current_cluster[-1]
            
            vertical_overlap = (min(region['bbox'][3], last_region['bbox'][3]) - 
                             max(region['bbox'][1], last_region['bbox'][1]))
            
            if vertical_overlap > -10:
                current_cluster.append(region)
            else:
                clusters.append(sorted(current_cluster, key=lambda x: x['bbox'][0]))
                current_cluster = [region]
        
        if current_cluster:
            clusters.append(sorted(current_cluster, key=lambda x: x['bbox'][0]))
        
        reading_order = []
        for cluster in clusters:
            reading_order.extend(cluster)
        
        return reading_order
    
    def analyze_page_structure(self, regions: Dict[str, List]) -> Dict[str, Any]:
        structure = {
            'has_header': len(regions['header']) > 0,
            'has_footer': len(regions['footer']) > 0,
            'has_sidebar': len(regions['sidebar']) > 0,
            'has_tables': len(regions['table']) > 0,
            'has_forms': len(regions['form']) > 0,
            'text_density': len(regions['text']),
            'layout_complexity': self.calculate_layout_complexity(regions)
        }
        
        return structure
    
    def calculate_layout_complexity(self, regions: Dict[str, List]) -> float:
        total_elements = sum(len(regions[key]) for key in regions)
        element_types = sum(1 for key in regions if len(regions[key]) > 0)
        
        complexity = (total_elements * element_types) / 100.0
        return min(complexity, 1.0)