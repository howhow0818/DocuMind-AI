import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig
from typing import Dict, Any

class DocumentTransformer(nn.Module):
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        model_name = config.get('model_name', 'microsoft/layoutlm-base-uncased')
        
        self.transformer = AutoModel.from_pretrained(model_name)
        self.hidden_size = self.transformer.config.hidden_size
        
        self.entity_classifier = nn.Linear(self.hidden_size, config.get('num_entity_labels', 10))
        self.relation_classifier = nn.Linear(self.hidden_size * 2, config.get('num_relation_types', 5))
        
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, input_ids, bbox, attention_mask=None):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        
        outputs = self.transformer(
            input_ids=input_ids,
            bbox=bbox,
            attention_mask=attention_mask
        )
        
        sequence_output = outputs.last_hidden_state
        sequence_output = self.dropout(sequence_output)
        
        entity_logits = self.entity_classifier(sequence_output)
        
        return {
            'entity_logits': entity_logits,
            'sequence_output': sequence_output
        }