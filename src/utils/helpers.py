"""
Helpers Module - Utility Functions
"""

import torch
import numpy as np
import random
import os
from PIL import Image
import torchvision.transforms as transforms


def set_seed(seed=42):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def get_device(use_mps=True):
    """Get the best available device."""
    if use_mps and torch.backends.mps.is_available():
        return torch.device('mps')
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


def preprocess_image(image):
    """Preprocess image for model input."""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    return transform(image)


def get_caption(detected_classes):
    """Generate caption based on detected classes."""
    if not detected_classes:
        return "Normal: No fracture, no metal"
    if "Fractura" in detected_classes and "Metal" in detected_classes:
        return "Fracture + Metal detected"
    if "Fractura" in detected_classes:
        return "Fracture detected"
    if "Metal" in detected_classes:
        return "Metal hardware detected"
    if "Texto" in detected_classes:
        return "Text annotations present"
    return "Abnormalities detected"


def get_treatment(detected_classes, probs):
    """
    Generate clinical description based on detected classes.
    This is a placeholder - actual implementation uses Ollama.
    """
    if not detected_classes:
        return "Normal wrist X-ray. No clinical intervention required."
    
    treatment_map = {
        "Fractura": "Immobilize wrist with splint or cast. Follow up with orthopedic specialist.",
        "Metal": "Monitor surgical hardware. Routine follow-up with orthopedic surgeon.",
        "Fractura_Metal": "Surgical consultation required. Possible internal fixation or hardware revision.",
        "Texto": "Text annotations present. Not clinically significant."
    }
    
    if "Fractura" in detected_classes and "Metal" in detected_classes:
        return treatment_map["Fractura_Metal"]
    elif "Fractura" in detected_classes:
        return treatment_map["Fractura"]
    elif "Metal" in detected_classes:
        return treatment_map["Metal"]
    else:
        return treatment_map["Texto"]


def get_class_names():
    """Get class names for the model."""
    return ['Fractura', 'Metal', 'Texto']


def count_parameters(model):
    """Count the number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
