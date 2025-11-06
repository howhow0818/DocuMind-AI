from pydantic import BaseModel
from typing import Dict, List, Any, Optional

class DocumentRequest(BaseModel):
    file_path: str
    options: Optional[Dict[str, Any]] = None

class DocumentResponse(BaseModel):
    document_type: Dict[str, Any]
    entities: List[Dict[str, Any]]
    tables: List[Dict[str, Any]]
    layout: Dict[str, Any]
    full_text: str
    processing_time: float
