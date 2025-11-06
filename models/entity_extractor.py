import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from typing import List, Dict, Any

class EntityExtractor(nn.Module):
    def __init__(self, num_entities: int, model_name: str = "bert-base-uncased"):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        self.entity_classifier = nn.Sequential(
            nn.Linear(self.bert.config.hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, num_entities)
        )
        
        self.entity_types = [
            'person', 'organization', 'date', 'amount', 'address',
            'invoice_number', 'total_amount', 'due_date', 'vendor', 'customer'
        ]
    
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state
        entity_logits = self.entity_classifier(sequence_output)
        return entity_logits
    
    def extract_entities(self, text: str) -> List[Dict[str, Any]]:
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, 
                              padding=True, max_length=512)
        
        with torch.no_grad():
            entity_logits = self.forward(inputs['input_ids'], inputs['attention_mask'])
            predictions = torch.argmax(entity_logits, dim=2)
            probabilities = torch.softmax(entity_logits, dim=2)
        
        tokens = self.tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
        entities = []
        current_entity = None
        
        for i, (token, pred, prob) in enumerate(zip(tokens, predictions[0], probabilities[0])):
            if token in ['[CLS]', '[SEP]', '[PAD]']:
                continue
            
            entity_type = self.entity_types[pred.item()]
            confidence = float(prob[pred.item()])
            
            if entity_type != 'O':
                if current_entity and current_entity['type'] == entity_type:
                    current_entity['text'] += ' ' + token
                    current_entity['end_pos'] = i
                    current_entity['confidence'] = min(current_entity['confidence'], confidence)
                else:
                    if current_entity:
                        entities.append(current_entity)
                    current_entity = {
                        'text': token,
                        'type': entity_type,
                        'start_pos': i,
                        'end_pos': i,
                        'confidence': confidence
                    }
            else:
                if current_entity:
                    entities.append(current_entity)
                    current_entity = None
        
        if current_entity:
            entities.append(current_entity)
        
        return entities