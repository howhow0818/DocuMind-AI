import cv2
import numpy as np
import pandas as pd
from typing import Dict, List, Any

class TableProcessor:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
    
    def extract_table_structure(self, image: np.ndarray, table_bbox: List[int]) -> Dict[str, Any]:
        x1, y1, x2, y2 = table_bbox
        table_roi = image[y1:y2, x1:x2]
        
        gray = cv2.cvtColor(table_roi, cv2.COLOR_BGR2GRAY)
        binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                     cv2.THRESH_BINARY, 11, 2)
        
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 50))
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (50, 1))
        
        vertical_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vertical_kernel, iterations=2)
        horizontal_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
        
        table_mask = cv2.addWeighted(vertical_lines, 0.5, horizontal_lines, 0.5, 0.0)
        table_mask = cv2.threshold(table_mask, 128, 255, cv2.THRESH_BINARY)[1]
        
        contours, _ = cv2.findContours(table_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        cells = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if w > 20 and h > 20:
                cells.append({
                    'bbox': [x + x1, y + y1, x + w + x1, y + h + y1],
                    'area': w * h
                })
        
        rows, cols = self.organize_cells_into_grid(cells)
        
        return {
            'cells': cells,
            'rows': rows,
            'columns': cols,
            'row_count': len(rows),
            'column_count': len(cols) if cols else 0
        }
    
    def organize_cells_into_grid(self, cells: List[Dict]) -> Tuple[List, List]:
        if not cells:
            return [], []
        
        y_coords = sorted(list(set(cell['bbox'][1] for cell in cells)))
        x_coords = sorted(list(set(cell['bbox'][0] for cell in cells)))
        
        rows = []
        for y in y_coords:
            row_cells = [cell for cell in cells if abs(cell['bbox'][1] - y) < 10]
            rows.append(sorted(row_cells, key=lambda x: x['bbox'][0]))
        
        cols = []
        for x in x_coords:
            col_cells = [cell for cell in cells if abs(cell['bbox'][0] - x) < 10]
            cols.append(sorted(col_cells, key=lambda x: x['bbox'][1]))
        
        return rows, cols
    
    def extract_table_data(self, image: np.ndarray, table_structure: Dict[str, Any], 
                         ocr_engine) -> pd.DataFrame:
        rows = table_structure['rows']
        
        table_data = []
        for row in rows:
            row_data = []
            for cell in row:
                cell_image = self.extract_cell_image(image, cell['bbox'])
                cell_text = ocr_engine.extract_full_text(cell_image)
                row_data.append(cell_text.strip())
            table_data.append(row_data)
        
        df = pd.DataFrame(table_data)
        
        if not df.empty:
            df = self.clean_dataframe(df)
        
        return df
    
    def extract_cell_image(self, image: np.ndarray, bbox: List[int]) -> np.ndarray:
        x1, y1, x2, y2 = bbox
        padding = 2
        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(image.shape[1], x2 + padding)
        y2 = min(image.shape[0], y2 + padding)
        
        return image[y1:y2, x1:x2]
    
    def clean_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.replace(r'^\s*$', np.nan, regex=True)
        df = df.dropna(how='all')
        df = df.dropna(axis=1, how='all')
        
        df = df.applymap(lambda x: re.sub(r'\s+', ' ', str(x)).strip() if pd.notna(x) else x)
        
        return df
