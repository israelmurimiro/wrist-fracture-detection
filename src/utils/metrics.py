"""
Metrics Module - Evaluation Metrics for Model Performance
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)


def calculate_metrics(y_true, y_pred, y_probs=None, class_names=None):
    """
    Calculate comprehensive classification metrics.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        y_probs: Predicted probabilities (optional)
        class_names: List of class names (optional)
    
    Returns:
        dict: Dictionary containing all metrics
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
        'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
        'f1': f1_score(y_true, y_pred, average='weighted', zero_division=0),
        'confusion_matrix': confusion_matrix(y_true, y_pred),
    }
    
    # Add per-class metrics
    metrics['per_class'] = {
        'precision': precision_score(y_true, y_pred, average=None, zero_division=0).tolist(),
        'recall': recall_score(y_true, y_pred, average=None, zero_division=0).tolist(),
        'f1': f1_score(y_true, y_pred, average=None, zero_division=0).tolist(),
    }
    
    if class_names:
        metrics['classification_report'] = classification_report(
            y_true, y_pred, target_names=class_names, zero_division=0
        )
    
    # Add AUC if probabilities provided
    if y_probs is not None:
        try:
            n_classes = y_probs.shape[1]
            aucs = []
            for i in range(n_classes):
                if len(np.unique(y_true[:, i])) > 1:
                    aucs.append(roc_auc_score(y_true[:, i], y_probs[:, i]))
                else:
                    aucs.append(float('nan'))
            metrics['auc_per_class'] = aucs
            metrics['auc_macro'] = np.nanmean(aucs)
        except:
            pass
    
    return metrics


def compute_auc(labels, probs):
    """Compute AUC for each class."""
    aucs = {}
    for i in range(labels.shape[1]):
        if len(np.unique(labels[:, i])) > 1:
            aucs[f'class_{i}'] = roc_auc_score(labels[:, i], probs[:, i])
    return aucs


def get_per_class_metrics(labels, preds, class_names):
    """Get per-class precision, recall, f1."""
    precision = precision_score(labels, preds, average=None, zero_division=0)
    recall = recall_score(labels, preds, average=None, zero_division=0)
    f1 = f1_score(labels, preds, average=None, zero_division=0)
    
    result = {}
    for i, name in enumerate(class_names):
        result[name] = {
            'precision': precision[i],
            'recall': recall[i],
            'f1': f1[i]
        }
    return result
