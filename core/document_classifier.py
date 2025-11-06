import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from typing import Dict, List, Any
import numpy as np

class DocumentClassifier(nn.Module):
    def __init__(self, num_classes: int, model_name: str = "bert-base-uncased"):
        super().__init__()
        self.num_classes = num_classes
        self.bert = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        self.classifier = nn.Sequential(
            nn.Linear(self.bert.config.hidden_size, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )
        
        self.document_types = [
            'invoice', 'contract', 'receipt', 'form', 'report', 
            'letter', 'manual', 'certificate', 'other'
        ]
    
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        logits = self.classifier(pooled_output)
        return logits
    
    def predict_document_type(self, text: str) -> Dict[str, Any]:
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, 
                              padding=True, max_length=512)
        
        with torch.no_grad():
            logits = self.forward(inputs['input_ids'], inputs['attention_mask'])
            probabilities = torch.softmax(logits, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
        
        return {
            'document_type': self.document_types[predicted_class],
            'confidence': float(probabilities[0][predicted_class]),
            'all_probabilities': {
                doc_type: float(prob) 
                for doc_type, prob in zip(self.document_types, probabilities[0])
            }
        }

class LayoutBasedClassifier:
    def __init__(self):
        self.feature_weights = {
            'table_density': 0.3,
            'form_density': 0.25,
            'text_density': 0.2,
            'structure_complexity': 0.25
        }
    
    def classify_by_layout(self, layout_analysis: Dict[str, Any]) -> Dict[str, Any]:
        features = self.extract_layout_features(layout_analysis)
        
        scores = {}
        
        scores['invoice'] = self.score_invoice(features)
        scores['contract'] = self.score_contract(features)
        scores['form'] = self.score_form(features)
        scores['report'] = self.score_report(features)
        
        best_type = max(scores, key=scores.get)
        confidence = scores[best_type] / sum(scores.values()) if sum(scores.values()) > 0 else 0
        
        return {
            'document_type': best_type,
            'confidence': confidence,
            'layout_scores': scores
        }
    
    def extract_layout_features(self, layout_analysis: Dict[str, Any]) -> Dict[str, float]:
        structure = layout_analysis['page_structure']
        
        return {
            'table_density': len(layout_analysis['tables']),
            'form_density': len(layout_analysis['forms']),
            'text_density': structure['text_density'] / 100.0,
            'structure_complexity': structure['layout_complexity'],
            'has_header': float(structure['has_header']),
            'has_footer': float(structure['has_footer'])
        }
    
    def score_invoice(self, features: Dict[str, float]) -> float:
        score = 0
        score += features['table_density'] * 0.4
        score += (1 - features['form_density']) * 0.3
        score += features['text_density'] * 0.2
        score += features['structure_complexity'] * 0.1
        return score
    
    def score_contract(self, features: Dict[str, float]) -> float:
        score = 0
        score += (1 - features['table_density']) * 0.3
        score += (1 - features['form_density']) * 0.2
        score += features['text_density'] * 0.4
        score += features['structure_complexity'] * 0.1
        return score
    
    def score_form(self, features: Dict[str, float]) -> float:
        score = 0
        score += features['form_density'] * 0.6
        score += features['table_density'] * 0.2
        score += (1 - features['text_density']) * 0.2
        return score
    
    def score_report(self, features: Dict[str, float]) -> float:
        score = 0
        score += features['table_density'] * 0.3
        score += features['text_density'] * 0.4
        score += features['structure_complexity'] * 0.3
        return score