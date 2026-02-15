"""Custom metrics definitions."""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def calculate_specificity(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate specificity (true negative rate).

    Args:
        y_true: True labels.
        y_pred: Predicted labels.

    Returns:
        Specificity score.
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return tn / (tn + fp) if (tn + fp) > 0 else 0.0


def calculate_npv(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate negative predictive value.

    Args:
        y_true: True labels.
        y_pred: Predicted labels.

    Returns:
        NPV score.
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return tn / (tn + fn) if (tn + fn) > 0 else 0.0


def calculate_all_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_pred_proba: np.ndarray | None = None,
) -> dict:
    """Calculate all standard classification metrics.

    Args:
        y_true: True labels.
        y_pred: Predicted labels.
        y_pred_proba: Predicted probabilities for positive class (optional).

    Returns:
        Dictionary of metric names to values.
    """
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "specificity": calculate_specificity(y_true, y_pred),
        "npv": calculate_npv(y_true, y_pred),
    }

    if y_pred_proba is not None:
        metrics["roc_auc"] = roc_auc_score(y_true, y_pred_proba)
        metrics["pr_auc"] = average_precision_score(y_true, y_pred_proba)

    return metrics


def calculate_medical_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_pred_proba: np.ndarray | None = None,
) -> dict:
    """Calculate metrics important for medical diagnosis.

    Focuses on recall (sensitivity) as missing a disease case is costly.

    Args:
        y_true: True labels.
        y_pred: Predicted labels.
        y_pred_proba: Predicted probabilities for positive class (optional).

    Returns:
        Dictionary of medical-focused metrics.
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    metrics = {
        "sensitivity": recall_score(y_true, y_pred, zero_division=0),
        "specificity": calculate_specificity(y_true, y_pred),
        "ppv": precision_score(y_true, y_pred, zero_division=0),
        "npv": calculate_npv(y_true, y_pred),
        "true_positives": int(tp),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "total_patients": len(y_true),
        "disease_prevalence": float(y_true.mean()),
    }

    if y_pred_proba is not None:
        metrics["roc_auc"] = roc_auc_score(y_true, y_pred_proba)

    return metrics


def calculate_confusion_matrix_dict(
    y_true: np.ndarray, y_pred: np.ndarray
) -> dict:
    """Calculate confusion matrix as a dictionary.

    Args:
        y_true: True labels.
        y_pred: Predicted labels.

    Returns:
        Dictionary with TP, TN, FP, FN counts.
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return {
        "true_positive": int(tp),
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
    }
