import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from core.document_classifier import DocumentClassifier
from models.entity_extractor import EntityExtractor
from utils.config import Config
from utils.visualization import DocumentVisualizer
import time

def train_document_classifier():
    config = Config()
    
    model = DocumentClassifier(num_classes=9)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    criterion = nn.CrossEntropyLoss()
    
    print("Starting document classifier training...")
    
    for epoch in range(config.get('training.epochs', 10)):
        total_loss = 0
        # Training loop would go here with actual data
        # This is a placeholder for the training logic
        
        if epoch % 5 == 0:
            print(f"Epoch {epoch}, Loss: {total_loss:.4f}")
    
    torch.save(model.state_dict(), 'models/document_classifier.pth')
    print("Training completed!")

def train_entity_extractor():
    config = Config()
    
    model = EntityExtractor(num_entities=10)
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)
    
    print("Starting entity extractor training...")
    
    for epoch in range(config.get('training.epochs', 15)):
        total_loss = 0
        # Training loop would go here with actual data
        
        if epoch % 5 == 0:
            print(f"Epoch {epoch}, Loss: {total_loss:.4f}")
    
    torch.save(model.state_dict(), 'models/entity_extractor.pth')
    print("Training completed!")

if __name__ == "__main__":
    train_document_classifier()
    train_entity_extractor()
