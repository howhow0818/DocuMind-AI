import torch
import torch.nn as nn
import torchvision.models as models
from typing import List, Dict, Any

class TableDetector(nn.Module):
    def __init__(self, num_classes: int = 2):
        super().__init__()
        self.backbone = models.resnet50(pretrained=True)
        self.backbone.fc = nn.Identity()
        
        self.bbox_regressor = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 4)
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        features = self.backbone(x)
        bbox = torch.sigmoid(self.bbox_regressor(features))
        class_logits = self.classifier(features)
        return bbox, class_logits