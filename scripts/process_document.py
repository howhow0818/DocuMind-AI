import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import cv2
import time
from core.ocr_engine import EnhancedOCREngine
from core.layout_analyzer import LayoutAnalyzer
from core.document_classifier import DocumentClassifier, LayoutBasedClassifier
from models.entity_extractor import EntityExtractor
from processing.preprocessor import DocumentPreprocessor
from processing.postprocessor import DocumentPostprocessor
from processing.table_processor import TableProcessor
from utils.config import Config
from utils.visualization import DocumentVisualizer
from utils.file_handlers import FileHandler

class DocumentProcessor:
    def __init__(self, config_path: str = "configs/default.yaml"):
        self.config = Config(config_path)
        
        self.ocr_engine = EnhancedOCREngine(self.config.get('ocr', {}))
        self.layout_analyzer = LayoutAnalyzer(self.config.get('layout', {}))
        self.document_classifier = DocumentClassifier(num_classes=9)
        self.layout_classifier = LayoutBasedClassifier()
        self.entity_extractor = EntityExtractor(num_entities=10)
        self.preprocessor = DocumentPreprocessor(self.config.get('preprocessing', {}))
        self.postprocessor = DocumentPostprocessor(self.config.get('postprocessing', {}))
        self.table_processor = TableProcessor(self.config.get('tables', {}))
        self.visualizer = DocumentVisualizer()
        self.file_handler = FileHandler()
        
        self.load_models()
    
    def load_models(self):
        try:
            self.document_classifier.load_state_dict(
                torch.load('models/document_classifier.pth', map_location='cpu')
            )
            self.entity_extractor.load_state_dict(
                torch.load('models/entity_extractor.pth', map_location='cpu')
            )
        except FileNotFoundError:
            print("Warning: Pre-trained models not found. Using untrained models.")
    
    def process_document(self, image_path: str) -> Dict[str, Any]:
        start_time = time.time()
        
        preprocessing_result = self.preprocessor.preprocess_document(image_path)
        processed_image = preprocessing_result['processed_image']
        
        ocr_result = self.ocr_engine.process_document(image_path)
        
        layout_analysis = self.layout_analyzer.analyze_document_layout(
            processed_image, ocr_result['text_regions']
        )
        
        text_based_classification = self.document_classifier.predict_document_type(
            ocr_result['full_text']
        )
        
        layout_based_classification = self.layout_classifier.classify_by_layout(
            layout_analysis
        )
        
        final_document_type = self.combine_classifications(
            text_based_classification, layout_based_classification
        )
        
        entities = self.entity_extractor.extract_entities(ocr_result['full_text'])
        validated_entities = self.postprocessor.validate_entities(entities)
        
        tables = {}
        for i, table in enumerate(layout_analysis['tables']):
            table_structure = self.table_processor.extract_table_structure(
                processed_image, table['bbox']
            )
            table_data = self.table_processor.extract_table_data(
                processed_image, table_structure, self.ocr_engine
            )
            tables[f"table_{i}"] = table_data
        
        layout_viz = self.visualizer.draw_layout_analysis(processed_image, layout_analysis)
        entities_viz = self.visualizer.draw_entities(processed_image, validated_entities)
        
        report_fig = self.visualizer.create_analysis_report({
            'layout_viz': layout_viz,
            'entities_viz': entities_viz,
            'document_type': final_document_type,
            'entities': validated_entities
        })
        
        processing_time = time.time() - start_time
        
        results = {
            'document_type': final_document_type,
            'entities': validated_entities,
            'tables': tables,
            'layout': layout_analysis,
            'full_text': self.postprocessor.clean_extracted_text(ocr_result['full_text']),
            'processing_time': processing_time,
            'visualizations': {
                'layout': layout_viz,
                'entities': entities_viz,
                'report': report_fig
            }
        }
        
        filename = os.path.splitext(os.path.basename(image_path))[0]
        self.file_handler.save_analysis_results(results, filename)
        self.file_handler.save_table_data(tables, filename)
        self.file_handler.save_visualizations(results['visualizations'], filename)
        
        return results
    
    def combine_classifications(self, text_based: Dict, layout_based: Dict) -> Dict[str, Any]:
        text_confidence = text_based['confidence']
        layout_confidence = layout_based['confidence']
        
        if text_confidence > layout_confidence:
            final_type = text_based['document_type']
            final_confidence = text_confidence
        else:
            final_type = layout_based['document_type']
            final_confidence = layout_confidence
        
        return {
            'document_type': final_type,
            'confidence': final_confidence,
            'text_based': text_based,
            'layout_based': layout_based
        }

def main():
    if len(sys.argv) != 2:
        print("Usage: python process_document.py <image_path>")
        sys.exit(1)
    
    image_path = sys.argv[1]
    processor = DocumentProcessor()
    
    print(f"Processing document: {image_path}")
    results = processor.process_document(image_path)
    
    print(f"\nDocument Analysis Results:")
    print(f"Document Type: {results['document_type']['document_type']} "
          f"(Confidence: {results['document_type']['confidence']:.2f})")
    print(f"Entities Found: {len(results['entities'])}")
    print(f"Tables Found: {len(results['tables'])}")
    print(f"Processing Time: {results['processing_time']:.2f}s")
    
    print(f"\nResults saved to output directory")

if __name__ == "__main__":
    main()