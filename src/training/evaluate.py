import torch
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score
from tqdm import tqdm

def evaluate_model(model, dataloader, device):
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Evaluating"):
            images = images.to(device)
            outputs = model(images)
            probs = torch.sigmoid(outputs).cpu().numpy()
            preds = (probs > 0.5).astype(int)
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())
            all_probs.extend(probs)
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    pred_labels = np.argmax(all_probs, axis=1)
    true_labels = np.argmax(all_labels, axis=1)
    return {
        'predictions': all_preds,
        'labels': all_labels,
        'probabilities': all_probs,
        'single_label_preds': pred_labels,
        'single_label_true': true_labels,
        'confusion_matrix': confusion_matrix(true_labels, pred_labels),
        'classification_report': classification_report(true_labels, pred_labels),
        'accuracy': (pred_labels == true_labels).mean()
    }

def compute_auc(all_labels, all_probs):
    aucs = {}
    for i in range(all_labels.shape[1]):
        if len(np.unique(all_labels[:, i])) > 1:
            aucs[f'class_{i}'] = roc_auc_score(all_labels[:, i], all_probs[:, i])
    return aucs
