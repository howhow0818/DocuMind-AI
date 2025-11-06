from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import uvicorn
from typing import List, Dict, Any
import tempfile
import os

class DocumentAPI:
    def __init__(self, document_processor):
        self.app = FastAPI(title="DocuMind AI API", 
                          description="Intelligent Document Processing System",
                          version="1.0.0")
        self.document_processor = document_processor
        self.setup_routes()
    
    def setup_routes(self):
        @self.app.post("/process-document/")
        async def process_document(file: UploadFile = File(...)):
            if not file.content_type.startswith('image/'):
                raise HTTPException(status_code=400, detail="File must be an image")
            
            with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as temp_file:
                content = await file.read()
                temp_file.write(content)
                temp_file_path = temp_file.name
            
            try:
                results = self.document_processor.process_document(temp_file_path)
                os.unlink(temp_file_path)
                return JSONResponse(content=results)
            except Exception as e:
                os.unlink(temp_file_path)
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/health/")
        async def health_check():
            return {"status": "healthy", "service": "DocuMind AI"}
    
    def run(self, host: str = "0.0.0.0", port: int = 8000):
        uvicorn.run(self.app, host=host, port=port)