import numpy as np
import warnings
import torch
from sklearn.metrics import (
    average_precision_score, roc_auc_score, f1_score,
    precision_score, recall_score
)
from sklearn.exceptions import UndefinedMetricWarning

# Globally suppress zero-division warnings for cleaner logs
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)


def _prep(x, ensure_2d=True):
    """Ensure input is a numpy array and handle basic shape requirements."""
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    arr = np.asarray(x)
    return arr[:, None] if (ensure_2d and arr.ndim == 1) else arr


def compute_binary_metrics(labels, outputs, threshold=0.5):
    """
    Compute core metrics using a FIXED threshold.
    Standard academic practice is using 0.5 unless otherwise justified.
    """
    y_true = _prep(labels)
    y_pred = (_prep(outputs) >= threshold).astype(int)

    metrics = {
        'f1': f1_score(y_true, y_pred, average='macro', zero_division=0),
        'prec': precision_score(y_true, y_pred, average='macro', zero_division=0),
        'recall': recall_score(y_true, y_pred, average='macro', zero_division=0),
        'acc': (y_true == y_pred).mean()
    }

    # Specificity = TN / (TN + FP) calculated for all classes
    spec_list = []
    for i in range(y_true.shape[1]):
        tn = ((y_true[:, i] == 0) & (y_pred[:, i] == 0)).sum()
        fp = ((y_true[:, i] == 0) & (y_pred[:, i] == 1)).sum()
        spec_list.append(tn / (tn + fp) if (tn + fp) > 0 else 0.0)

    metrics['spec'] = np.mean(spec_list)
    return metrics


def compute_auprc(labels, outputs):
    """
    Compute Macro AUPRC for all classes.
    Uses standard sklearn macro-averaging without custom class skipping.
    """
    y_true, y_score = _prep(labels), _prep(outputs)
    # Using standard library's macro average directly
    return float(average_precision_score(y_true, y_score, average='macro'))


def compute_auroc(labels, outputs):
    """
    Compute Macro AUROC for all classes.
    Note: Will raise error if a class has only one unique label,
    forcing user to acknowledge data imbalance.
    """
    y_true, y_score = _prep(labels), _prep(outputs)
    return float(roc_auc_score(y_true, y_score, average='macro'))


def compute_confusion_matrices(labels, outputs, normalize=False):
    """Standard multi-label confusion matrices: [C, 2, 2]."""
    y_t, y_p = (_prep(labels) >= 0.5).astype(int), (_prep(outputs) >= 0.5).astype(int)
    C = y_t.shape[1]
    matrices = np.zeros((C, 2, 2))

    for c in range(C):
        yt, yp = y_t[:, c], y_p[:, c]
        matrices[c] = [[((yt == 0) & (yp == 0)).sum(), ((yt == 1) & (yp == 0)).sum()],
                       [((yt == 0) & (yp == 1)).sum(), ((yt == 1) & (yp == 1)).sum()]]

    if normalize:
        sums = matrices.sum(axis=(1, 2), keepdims=True)
        matrices = np.divide(matrices, sums, out=np.zeros_like(matrices), where=sums != 0)
    return matrices


# --- Academic Standard Wrappers ---

def compute_f1_sp(l, o):
    m = compute_binary_metrics(l, o)
    return m['f1'], m['spec']


def compute_accuracy(l, o):
    return compute_binary_metrics(l, o)['acc']


def compute_precision_recall(l, o):
    m = compute_binary_metrics(l, o)
    return m['prec'], m['recall']