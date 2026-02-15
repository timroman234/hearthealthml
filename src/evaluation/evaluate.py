"""Model evaluation routines."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.base import BaseEstimator
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

from src.evaluation.metrics import calculate_all_metrics, calculate_medical_metrics
from src.utils.logger import get_logger

logger = get_logger(__name__)


def evaluate_model(
    model: BaseEstimator,
    X_test: np.ndarray,
    y_test: np.ndarray,
    threshold: float = 0.5,
) -> dict:
    """Compute all evaluation metrics.

    Args:
        model: Trained model.
        X_test: Test features.
        y_test: Test labels.
        threshold: Classification threshold.

    Returns:
        Dictionary of all metrics.
    """
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_proba >= threshold).astype(int)

    metrics = calculate_all_metrics(y_test, y_pred, y_pred_proba)
    metrics["threshold"] = threshold
    metrics["confusion_matrix"] = confusion_matrix(y_test, y_pred).tolist()
    metrics["classification_report"] = classification_report(
        y_test, y_pred, output_dict=True
    )

    logger.info(f"Evaluation metrics: accuracy={metrics['accuracy']:.4f}, "
                f"roc_auc={metrics.get('roc_auc', 'N/A'):.4f}")

    return metrics


def evaluate_medical(
    model: BaseEstimator,
    X_test: np.ndarray,
    y_test: np.ndarray,
    threshold: float = 0.5,
) -> dict:
    """Compute medical-focused evaluation metrics.

    Args:
        model: Trained model.
        X_test: Test features.
        y_test: Test labels.
        threshold: Classification threshold.

    Returns:
        Dictionary of medical metrics.
    """
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_proba >= threshold).astype(int)

    metrics = calculate_medical_metrics(y_test, y_pred, y_pred_proba)

    logger.info(f"Medical metrics: sensitivity={metrics['sensitivity']:.4f}, "
                f"specificity={metrics['specificity']:.4f}")

    return metrics


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_path: Path | None = None,
    labels: list[str] | None = None,
) -> plt.Figure:
    """Generate confusion matrix plot.

    Args:
        y_true: True labels.
        y_pred: Predicted labels.
        output_path: Path to save plot (optional).
        labels: Class labels for display.

    Returns:
        Matplotlib figure.
    """
    if labels is None:
        labels = ["Healthy", "Heart Disease"]

    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved confusion matrix to {output_path}")

    return fig


def plot_roc_curve(
    model: BaseEstimator,
    X_test: np.ndarray,
    y_test: np.ndarray,
    output_path: Path | None = None,
) -> plt.Figure:
    """Generate ROC curve plot.

    Args:
        model: Trained model.
        X_test: Test features.
        y_test: Test labels.
        output_path: Path to save plot (optional).

    Returns:
        Matplotlib figure.
    """
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    auc = roc_auc_score(y_test, y_pred_proba)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, label=f"ROC Curve (AUC = {auc:.3f})", linewidth=2)
    ax.plot([0, 1], [0, 1], "k--", label="Random Classifier")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved ROC curve to {output_path}")

    return fig


def plot_precision_recall_curve(
    model: BaseEstimator,
    X_test: np.ndarray,
    y_test: np.ndarray,
    output_path: Path | None = None,
) -> plt.Figure:
    """Generate Precision-Recall curve plot.

    Args:
        model: Trained model.
        X_test: Test features.
        y_test: Test labels.
        output_path: Path to save plot (optional).

    Returns:
        Matplotlib figure.
    """
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(recall, precision, linewidth=2)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve")
    ax.grid(True, alpha=0.3)

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved PR curve to {output_path}")

    return fig


def plot_calibration_curve(
    model: BaseEstimator,
    X_test: np.ndarray,
    y_test: np.ndarray,
    n_bins: int = 10,
    output_path: Path | None = None,
) -> plt.Figure:
    """Generate calibration curve plot.

    Args:
        model: Trained model.
        X_test: Test features.
        y_test: Test labels.
        n_bins: Number of bins for calibration.
        output_path: Path to save plot (optional).

    Returns:
        Matplotlib figure.
    """
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    prob_true, prob_pred = calibration_curve(y_test, y_pred_proba, n_bins=n_bins)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(prob_pred, prob_true, "s-", label="Model")
    ax.plot([0, 1], [0, 1], "k--", label="Perfectly Calibrated")
    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Fraction of Positives")
    ax.set_title("Calibration Curve")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved calibration curve to {output_path}")

    return fig


def find_optimal_threshold(
    model: BaseEstimator,
    X_val: np.ndarray,
    y_val: np.ndarray,
    optimize_for: str = "f1",
) -> float:
    """Find optimal classification threshold.

    Args:
        model: Trained model.
        X_val: Validation features.
        y_val: Validation labels.
        optimize_for: Metric to optimize ('f1', 'recall', 'precision', 'youden').

    Returns:
        Optimal threshold value.
    """
    from sklearn.metrics import f1_score, precision_score, recall_score

    y_pred_proba = model.predict_proba(X_val)[:, 1]
    thresholds = np.arange(0.1, 0.9, 0.01)
    scores = []

    for threshold in thresholds:
        y_pred = (y_pred_proba >= threshold).astype(int)

        if optimize_for == "f1":
            score = f1_score(y_val, y_pred, zero_division=0)
        elif optimize_for == "recall":
            score = recall_score(y_val, y_pred, zero_division=0)
        elif optimize_for == "precision":
            score = precision_score(y_val, y_pred, zero_division=0)
        elif optimize_for == "youden":
            # Youden's J statistic: sensitivity + specificity - 1
            tn, fp, fn, tp = confusion_matrix(y_val, y_pred).ravel()
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            score = sensitivity + specificity - 1
        else:
            raise ValueError(f"Unknown metric: {optimize_for}")

        scores.append(score)

    optimal_idx = np.argmax(scores)
    optimal_threshold = thresholds[optimal_idx]

    logger.info(
        f"Optimal threshold for {optimize_for}: {optimal_threshold:.2f} "
        f"(score: {scores[optimal_idx]:.4f})"
    )

    return float(optimal_threshold)
