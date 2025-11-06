import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from core.ocr_engine import EnhancedOCREngine
from core.document_classifier import DocumentClassifier
from models.entity_extractor import EntityExtractor
from utils.config import Config
import json

def evaluate_ocr_accuracy(test_data_path: str):
    config = Config()
    ocr_engine = EnhancedOCREngine(config.get('ocr', {}))
    
    with open(test_data_path, 'r') as f:
        test_data = json.load(f)
    
    total_accuracy = 0
    total_samples = len(test_data)
    
    for sample in test_data:
        image_path = sample['image_path']
        expected_text = sample['expected_text']
        
        result = ocr_engine.process_document(image_path)
        extracted_text = result['full_text']
        
        accuracy = calculate_text_similarity(expected_text, extracted_text)
        total_accuracy += accuracy
    
    avg_accuracy = total_accuracy / total_samples
    print(f"OCR Accuracy: {avg_accuracy:.4f}")
    
    return avg_accuracy

def calculate_text_similarity(text1: str, text2: str) -> float:
    import difflib
    return difflib.SequenceMatcher(None, text1.lower(), text2.lower()).ratio()

def evaluate_document_classification(test_data_path: str):
    config = Config()
    classifier = DocumentClassifier(num_classes=9)
    
    try:
        classifier.load_state_dict(torch.load('models/document_classifier.pth'))
    except FileNotFoundError:
        print("Document classifier model not found")
        return
    
    with open(test_data_path, 'r') as f:
        test_data = json.load(f)
    
    correct_predictions = 0
    total_samples = len(test_data)
    
    for sample in test_data:
        text = sample['text']
        expected_type = sample['document_type']
        
        prediction = classifier.predict_document_type(text)
        if prediction['document_type'] == expected_type:
            correct_predictions += 1
    
    accuracy = correct_predictions / total_samples
    print(f"Document Classification Accuracy: {accuracy:.4f}")
    
    return accuracy

def main():
    if len(sys.argv) != 2:
        print("Usage: python evaluate.py <test_data_directory>")
        sys.exit(1)
    
    test_dir = sys.argv[1]
    
    ocr_test_path = os.path.join(test_dir, "ocr_test_data.json")
    classification_test_path = os.path.join(test_dir, "classification_test_data.json")
    
    if os.path.exists(ocr_test_path):
        evaluate_ocr_accuracy(ocr_test_path)
    
    if os.path.exists(classification_test_path):
        evaluate_document_classification(classification_test_path)

if __name__ == "__main__":
    main()