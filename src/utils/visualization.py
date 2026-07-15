"""
Visualization Module - Plotting Utilities
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import roc_curve, auc, confusion_matrix


def plot_confusion_matrix(y_true, y_pred, class_names=None, title="Confusion Matrix"):
    """Plot confusion matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(title)
    plt.tight_layout()
    return plt.gcf()


def plot_roc_curves(y_true, y_probs, class_names=None, title="ROC Curves"):
    """Plot ROC curves for each class."""
    plt.figure(figsize=(8, 6))
    
    n_classes = y_probs.shape[1]
    colors = plt.cm.Set3(np.linspace(0, 1, n_classes))
    
    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_true[:, i], y_probs[:, i])
        roc_auc = auc(fpr, tpr)
        label = f"{class_names[i] if class_names else f'Class {i}'} (AUC = {roc_auc:.3f})"
        plt.plot(fpr, tpr, color=colors[i], label=label)
    
    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    return plt.gcf()


def plot_training_history(train_losses, val_losses, val_accuracies):
    """Plot training loss and accuracy curves."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    axes[0].plot(range(1, len(train_losses) + 1), train_losses, 'b-', label='Train Loss')
    axes[0].plot(range(1, len(val_losses) + 1), val_losses, 'r-', label='Validation Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Loss Over Epochs')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(range(1, len(val_accuracies) + 1), val_accuracies, 'g-', label='Validation Accuracy')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Validation Accuracy Over Epochs')
    axes[1].legend()
    axes[1].set_ylim(0, 1)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_precision_recall_curve(y_true, y_probs, class_names=None):
    """Plot Precision-Recall curves for each class."""
    from sklearn.metrics import precision_recall_curve
    
    plt.figure(figsize=(8, 6))
    
    n_classes = y_probs.shape[1]
    colors = plt.cm.Set3(np.linspace(0, 1, n_classes))
    
    for i in range(n_classes):
        precision, recall, _ = precision_recall_curve(y_true[:, i], y_probs[:, i])
        label = class_names[i] if class_names else f'Class {i}'
        plt.plot(recall, precision, color=colors[i], label=label)
    
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curves')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    return plt.gcf()


def plot_per_class_metrics(metrics_dict, class_names):
    """Plot bar chart of per-class precision, recall, f1."""
    x = np.arange(len(class_names))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    precision = [metrics_dict[name]['precision'] for name in class_names]
    recall = [metrics_dict[name]['recall'] for name in class_names]
    f1 = [metrics_dict[name]['f1'] for name in class_names]
    
    rects1 = ax.bar(x - width, precision, width, label='Precision')
    rects2 = ax.bar(x, recall, width, label='Recall')
    rects3 = ax.bar(x + width, f1, width, label='F1')
    
    ax.set_xlabel('Class')
    ax.set_ylabel('Score')
    ax.set_title('Per-Class Performance Metrics')
    ax.set_xticks(x)
    ax.set_xticklabels(class_names)
    ax.legend()
    ax.set_ylim(0, 1.05)
    
    # Add value labels
    for rects in [rects1, rects2, rects3]:
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.2f}',
                       xy=(rect.get_x() + rect.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    return fig
