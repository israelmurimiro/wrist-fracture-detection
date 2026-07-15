"""
Utils Tests - Unit tests for helper functions
"""

import pytest
import torch
import numpy as np
from PIL import Image

from src.utils.helpers import (
    set_seed,
    get_device,
    preprocess_image,
    get_caption,
    get_treatment,
    get_class_names,
    count_parameters,
)
from src.utils.metrics import calculate_metrics, compute_auc, get_per_class_metrics


def test_set_seed():
    """Test random seed setting."""
    set_seed(42)
    a = np.random.randn(10)
    set_seed(42)
    b = np.random.randn(10)
    np.testing.assert_array_equal(a, b)


def test_get_device():
    """Test device detection."""
    device = get_device()
    assert device is not None
    assert isinstance(device, torch.device)


def test_get_class_names():
    """Test class names retrieval."""
    names = get_class_names()
    assert len(names) == 3
    assert names == ['Fractura', 'Metal', 'Texto']


def test_get_caption():
    """Test caption generation for different scenarios."""
    # Normal case
    caption = get_caption([])
    assert "Normal" in caption
    
    # Fractura only
    caption = get_caption(["Fractura"])
    assert "Fracture" in caption
    
    # Metal only
    caption = get_caption(["Metal"])
    assert "Metal" in caption
    
    # Both
    caption = get_caption(["Fractura", "Metal"])
    assert "Fracture" in caption
    assert "Metal" in caption
    
    # Texto only
    caption = get_caption(["Texto"])
    assert "Text" in caption


def test_get_treatment():
    """Test treatment recommendation generation."""
    # Normal case
    treatment = get_treatment([], [0.1, 0.1, 0.8])
    assert "Normal" in treatment
    
    # Fractura
    treatment = get_treatment(["Fractura"], [0.9, 0.1, 0.1])
    assert "Immobilize" in treatment or "orthopedic" in treatment.lower()


def test_preprocess_image():
    """Test image preprocessing."""
    # Create dummy image
    img = Image.new('RGB', (224, 224), color='white')
    tensor = preprocess_image(img)
    
    assert tensor.shape == (3, 224, 224)
    assert isinstance(tensor, torch.Tensor)


def test_count_parameters():
    """Test parameter counting."""
    model = torch.nn.Linear(10, 5)
    count = count_parameters(model)
    assert count == 55  # 10*5 + 5 biases


def test_calculate_metrics():
    """Test metrics calculation."""
    y_true = np.array([0, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 0])
    
    metrics = calculate_metrics(y_true, y_pred)
    
    assert 'accuracy' in metrics
    assert 'precision' in metrics
    assert 'recall' in metrics
    assert 'f1' in metrics
    assert 'confusion_matrix' in metrics


def test_compute_auc():
    """Test AUC computation."""
    labels = np.array([[1, 0], [0, 1], [1, 0], [0, 1]])
    probs = np.array([[0.9, 0.1], [0.1, 0.9], [0.8, 0.2], [0.2, 0.8]])
    
    aucs = compute_auc(labels, probs)
    assert len(aucs) == 2


def test_get_per_class_metrics():
    """Test per-class metrics extraction."""
    y_true = np.array([0, 1, 0, 1, 0, 2])
    y_pred = np.array([0, 1, 0, 0, 0, 2])
    class_names = ['Fractura', 'Metal', 'Texto']
    
    metrics = get_per_class_metrics(y_true, y_pred, class_names)
    
    assert 'Fractura' in metrics
    assert 'Metal' in metrics
    assert 'Texto' in metrics
    assert 'precision' in metrics['Fractura']
    assert 'recall' in metrics['Fractura']
    assert 'f1' in metrics['Fractura']
