"""
Classifier Tests - Unit tests for model loading and prediction
"""

import pytest
import torch
from pathlib import Path
from PIL import Image
import numpy as np

from src.models.classifier import MultiLabelResNet, load_model, CLASS_NAMES
from src.utils.helpers import get_device, preprocess_image


def test_class_names():
    """Test class names are correct."""
    assert len(CLASS_NAMES) == 3
    assert "Fractura" in CLASS_NAMES
    assert "Metal" in CLASS_NAMES
    assert "Texto" in CLASS_NAMES


def test_model_initialization():
    """Test model can be initialized."""
    model = MultiLabelResNet(num_classes=3)
    assert model is not None
    assert hasattr(model, "backbone")
    assert hasattr(model, "class_names")


def test_model_forward():
    """Test model forward pass."""
    model = MultiLabelResNet(num_classes=3)
    model.eval()
    
    # Create dummy input
    dummy_input = torch.randn(1, 3, 224, 224)
    
    with torch.no_grad():
        output = model(dummy_input)
    
    assert output.shape == (1, 3)


def test_model_predict():
    """Test model predict method."""
    model = MultiLabelResNet(num_classes=3)
    model.eval()
    
    dummy_input = torch.randn(1, 3, 224, 224)
    probs, pred_class, confidence = model.predict(dummy_input)
    
    assert len(probs) == 3
    assert pred_class in CLASS_NAMES
    assert 0 <= confidence <= 1
    assert isinstance(confidence, float)


def test_load_model():
    """Test model loading from checkpoint."""
    model_path = Path("models/checkpoints/baseline_resnet50.pth")
    
    if model_path.exists():
        device = get_device()
        model = load_model(model_path, device)
        assert model is not None
        assert hasattr(model, "backbone")
    else:
        pytest.skip("Model checkpoint not found")
