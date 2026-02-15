"""Feature selection methods."""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.feature_selection import RFE, VarianceThreshold

from src.utils.logger import get_logger

logger = get_logger(__name__)


def remove_correlated_features(
    df: pd.DataFrame, threshold: float = 0.9
) -> list[str]:
    """Identify features to remove due to high correlation.

    Args:
        df: DataFrame with features (numeric only).
        threshold: Correlation threshold above which to remove.

    Returns:
        List of column names to drop.
    """
    # Select only numeric columns
    numeric_df = df.select_dtypes(include=[np.number])

    corr_matrix = numeric_df.corr().abs()
    upper = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )

    to_drop = [col for col in upper.columns if any(upper[col] > threshold)]

    if to_drop:
        logger.info(f"Features to remove due to high correlation: {to_drop}")
    else:
        logger.info(f"No features exceed correlation threshold of {threshold}")

    return to_drop


def select_by_variance(
    df: pd.DataFrame, threshold: float = 0.01
) -> list[str]:
    """Select features with variance above threshold.

    Args:
        df: DataFrame with features (numeric only).
        threshold: Minimum variance to keep.

    Returns:
        List of selected column names.
    """
    numeric_df = df.select_dtypes(include=[np.number])

    selector = VarianceThreshold(threshold=threshold)
    selector.fit(numeric_df)

    selected = numeric_df.columns[selector.get_support()].tolist()
    removed = set(numeric_df.columns) - set(selected)

    if removed:
        logger.info(f"Removed low variance features: {removed}")

    logger.info(f"Selected {len(selected)} features by variance threshold")
    return selected


def select_by_rfe(
    estimator: BaseEstimator,
    X: np.ndarray,
    y: np.ndarray,
    n_features: int = 10,
    step: int = 1,
) -> list[int]:
    """Select top N features using Recursive Feature Elimination.

    Args:
        estimator: Estimator with fit method and coef_ or feature_importances_.
        X: Feature matrix.
        y: Target vector.
        n_features: Number of features to select.
        step: Number of features to remove at each iteration.

    Returns:
        List of selected feature indices.
    """
    rfe = RFE(estimator, n_features_to_select=n_features, step=step)
    rfe.fit(X, y)

    selected_indices = np.where(rfe.support_)[0].tolist()
    logger.info(f"RFE selected {len(selected_indices)} features")

    return selected_indices


def select_by_importance(
    model: BaseEstimator,
    X: np.ndarray,
    y: np.ndarray,
    threshold: float = 0.01,
) -> list[int]:
    """Select features with importance above threshold.

    Args:
        model: Fitted model with coef_ or feature_importances_.
        X: Feature matrix.
        y: Target vector.
        threshold: Minimum importance to keep.

    Returns:
        List of selected feature indices.
    """
    model.fit(X, y)

    if hasattr(model, "coef_"):
        importances = np.abs(model.coef_).flatten()
    elif hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    else:
        raise ValueError("Model must have coef_ or feature_importances_ attribute")

    # Normalize importances
    importances = importances / importances.sum()

    selected_indices = np.where(importances > threshold)[0].tolist()
    logger.info(
        f"Selected {len(selected_indices)} features by importance threshold {threshold}"
    )

    return selected_indices


def get_feature_importances(
    model: BaseEstimator, feature_names: list[str]
) -> pd.DataFrame:
    """Get feature importances as a DataFrame.

    Args:
        model: Fitted model with coef_ or feature_importances_.
        feature_names: List of feature names.

    Returns:
        DataFrame with feature names and importances, sorted descending.
    """
    if hasattr(model, "coef_"):
        importances = np.abs(model.coef_).flatten()
    elif hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    else:
        raise ValueError("Model must have coef_ or feature_importances_ attribute")

    df = pd.DataFrame({
        "feature": feature_names,
        "importance": importances,
    })
    df = df.sort_values("importance", ascending=False).reset_index(drop=True)

    return df
