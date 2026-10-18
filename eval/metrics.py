"""Metrics the referee reports.

Part of the referee. Agents must not edit anything under eval/ (see AGENTS.md).
"""

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score


def accuracy(y_true, y_pred):
    return float(accuracy_score(y_true, y_pred))


def macro_f1(y_true, y_pred, labels):
    """Macro-F1 over every label in the dataset, so a class the model never predicts counts as 0."""
    return float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0))


def confusion(y_true, y_pred, labels):
    return confusion_matrix(y_true, y_pred, labels=labels).tolist()


def expected_calibration_error(confidence, correct, n_bins=10):
    """ECE: how far stated confidence is from actual accuracy, weighted by bin size."""
    confidence = np.asarray(confidence, dtype=float)
    correct = np.asarray(correct, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    # Right-closed bins so confidence 1.0 lands in the last bin.
    bins = np.clip(np.digitize(confidence, edges[1:-1], right=True), 0, n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        mask = bins == b
        if mask.any():
            ece += mask.mean() * abs(correct[mask].mean() - confidence[mask].mean())
    return float(ece)


def top_confidence(proba, classes, y_true):
    """Confidence and correctness of the argmax class, per sample."""
    top = proba.argmax(axis=1)
    return proba.max(axis=1), classes[top] == y_true
# guard test
