"""
Classifier Module - ResNet-50 Multi-Label Classifier
"""

import torch
import torch.nn as nn
from torchvision.models import resnet50
import numpy as np

CLASS_NAMES = ['Fractura', 'Metal', 'Texto']


class MultiLabelResNet(nn.Module):
    """ResNet-50 modified for multi-label classification."""
    
    def __init__(self, num_classes=3):
        super(MultiLabelResNet, self).__init__()
        self.backbone = resnet50(weights=None)
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)
        self.class_names = CLASS_NAMES
    
    def forward(self, x):
        return self.backbone(x)
    
    def predict(self, x):
        with torch.no_grad():
            outputs = self(x)
            probs = torch.sigmoid(outputs).cpu().numpy()[0]
        
        pred_idx = int(np.argmax(probs))
        pred_class = self.class_names[pred_idx]
        confidence = float(probs[pred_idx])
        
        return probs, pred_class, confidence


def load_model(model_path, device):
    model = MultiLabelResNet(num_classes=3)
    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    return model
