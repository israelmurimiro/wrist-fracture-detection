"""
Wrist Fracture Detection - Source Package
"""

from .data import FractureDataset
from .models import MultiLabelResNet, load_model, CLASS_NAMES
from .utils import (
    set_seed,
    get_device,
    preprocess_image,
    get_caption,
    get_treatment,
    get_class_names,
    count_parameters,
    calculate_metrics,
    compute_auc,
    get_per_class_metrics,
    plot_confusion_matrix,
    plot_roc_curves,
    plot_training_history,
    plot_precision_recall_curve,
    plot_per_class_metrics
)

__version__ = "1.0.0"

__all__ = [
    # Data
    "FractureDataset",
    # Models
    "MultiLabelResNet",
    "load_model",
    "CLASS_NAMES",
    # Utils
    "set_seed",
    "get_device",
    "preprocess_image",
    "get_caption",
    "get_treatment",
    "get_class_names",
    "count_parameters",
    "calculate_metrics",
    "compute_auc",
    "get_per_class_metrics",
    "plot_confusion_matrix",
    "plot_roc_curves",
    "plot_training_history",
    "plot_precision_recall_curve",
    "plot_per_class_metrics",
]
