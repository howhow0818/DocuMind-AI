import re
from typing import Dict, List, Any
from datetime import datetime

class DocumentPostprocessor:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
    def clean_extracted_text(self, text: str) -> str:
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^\w\s@.,$€£¥%()-]', '', text)
        text = text.strip()
        
        return text
    
    def validate_entities(self, entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        valid_entities = []
        
        for entity in entities:
            if self.is_valid_entity(entity):
                entity['normalized'] = self.normalize_entity_value(entity)
                valid_entities.append(entity)
        
        return valid_entities
    
    def is_valid_entity(self, entity: Dict[str, Any]) -> bool:
        min_confidence = self.config.get('min_entity_confidence', 0.7)
        
        if entity['confidence'] < min_confidence:
            return False
        
        text = entity.get('text', '').strip()
        if len(text) < 2:
            return False
        
        entity_type = entity.get('type', '')
        if entity_type in ['amount', 'date']:
            return self.validate_structured_entity(entity_type, text)
        
        return True
    
    def validate_structured_entity(self, entity_type: str, value: str) -> bool:
        if entity_type == 'amount':
            amount_pattern = r'^[$€£¥]?\s*\d{1,3}(?:,\d{3})*(?:\.\d{2})?$'
            return bool(re.match(amount_pattern, value.strip()))
        
        elif entity_type == 'date':
            date_patterns = [
                r'\d{1,2}/\d{1,2}/\d{4}',
                r'\d{4}-\d{1,2}-\d{1,2}',
                r'\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{4}'
            ]
            return any(re.search(pattern, value) for pattern in date_patterns)
        
        return True
    
    def normalize_entity_value(self, entity: Dict[str, Any]) -> str:
        entity_type = entity.get('type', '')
        value = entity.get('text', '')
        
        if entity_type == 'amount':
            return self.normalize_amount(value)
        elif entity_type == 'date':
            return self.normalize_date(value)
        elif entity_type in ['person', 'organization']:
            return value.title()
        
        return value
    
    def normalize_amount(self, amount_str: str) -> str:
        amount_str = re.sub(r'[^\d.,]', '', amount_str)
        
        if ',' in amount_str and '.' in amount_str:
            if amount_str.rfind(',') > amount_str.rfind('.'):
                amount_str = amount_str.replace('.', '').replace(',', '.')
            else:
                amount_str = amount_str.replace(',', '')
        
        elif ',' in amount_str:
            if len(amount_str.split(',')[-1]) == 3:
                amount_str = amount_str.replace(',', '')
            else:
                amount_str = amount_str.replace(',', '.')
        
        try:
            amount = float(amount_str)
            return f"{amount:.2f}"
        except ValueError:
            return amount_str
    
    def normalize_date(self, date_str: str) -> str:
        date_patterns = [
            (r'(\d{1,2})/(\d{1,2})/(\d{4})', '%m/%d/%Y'),
            (r'(\d{4})-(\d{1,2})-(\d{1,2})', '%Y-%m-%d'),
            (r'(\d{1,2})\s+(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+(\d{4})', '%d %b %Y')
        ]
        
        for pattern, date_format in date_patterns:
            match = re.search(pattern, date_str)
            if match:
                try:
                    if date_format == '%d %b %Y':
                        month_map = {
                            'Jan': '01', 'Feb': '02', 'Mar': '03', 'Apr': '04',
                            'May': '05', 'Jun': '06', 'Jul': '07', 'Aug': '08',
                            'Sep': '09', 'Oct': '10', 'Nov': '11', 'Dec': '12'
                        }
                        day, month, year = match.groups()
                        normalized = f"{year}-{month_map[month]}-{day.zfill(2)}"
                    else:
                        normalized = datetime.strptime(match.group(), 
                                                     date_format.replace('%', '')).strftime('%Y-%m-%d')
                    return normalized
                except ValueError:
                    continue
        
        return date_str