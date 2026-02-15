"""Inference logic."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer

from src.models.registry import ModelRegistry
from src.utils.logger import get_logger

logger = get_logger(__name__)


def load_production_model(
    model_name: str = "logistic_regression",
    version: str = "latest",
    registry_path: Path | None = None,
) -> tuple[BaseEstimator, ColumnTransformer]:
    """Load latest production model and preprocessor.

    Args:
        model_name: Name of the model to load.
        version: Version string or 'latest'.
        registry_path: Path to model registry. Uses default if None.

    Returns:
        Tuple of (model, preprocessor).
    """
    if registry_path is None:
        registry_path = Path("models")

    registry = ModelRegistry(registry_path)
    model, preprocessor = registry.load_model(model_name, version)

    return model, preprocessor


def predict_single(
    model: BaseEstimator,
    preprocessor: ColumnTransformer,
    patient_data: dict,
    threshold: float = 0.5,
) -> dict:
    """Make prediction for a single patient.

    Args:
        model: Trained model.
        preprocessor: Fitted preprocessor.
        patient_data: Dictionary of patient features.
        threshold: Classification threshold.

    Returns:
        Dictionary with prediction, probability, and risk_level.
    """
    df = pd.DataFrame([patient_data])
    X = preprocessor.transform(df)

    proba = model.predict_proba(X)[0, 1]
    prediction = int(proba >= threshold)

    # Determine risk level
    if proba >= 0.7:
        risk_level = "high"
    elif proba >= 0.3:
        risk_level = "medium"
    else:
        risk_level = "low"

    result = {
        "prediction": prediction,
        "prediction_label": "heart_disease" if prediction == 1 else "healthy",
        "probability": float(proba),
        "risk_level": risk_level,
        "threshold_used": threshold,
    }

    logger.info(
        f"Prediction: {result['prediction_label']} "
        f"(probability: {result['probability']:.2%}, risk: {result['risk_level']})"
    )

    return result


def predict_batch(
    model: BaseEstimator,
    preprocessor: ColumnTransformer,
    df: pd.DataFrame,
    threshold: float = 0.5,
) -> pd.DataFrame:
    """Make predictions for multiple patients.

    Args:
        model: Trained model.
        preprocessor: Fitted preprocessor.
        df: DataFrame with patient features.
        threshold: Classification threshold.

    Returns:
        DataFrame with added prediction columns.
    """
    X = preprocessor.transform(df)
    probas = model.predict_proba(X)[:, 1]
    predictions = (probas >= threshold).astype(int)

    results = df.copy()
    results["prediction"] = predictions
    results["probability"] = probas
    results["prediction_label"] = np.where(
        predictions == 1, "heart_disease", "healthy"
    )
    results["risk_level"] = pd.cut(
        probas,
        bins=[0, 0.3, 0.7, 1.0],
        labels=["low", "medium", "high"],
    )

    n_positive = predictions.sum()
    logger.info(
        f"Batch prediction: {len(df)} patients, "
        f"{n_positive} ({n_positive/len(df):.1%}) predicted positive"
    )

    return results


def predict_proba(
    model: BaseEstimator,
    preprocessor: ColumnTransformer,
    df: pd.DataFrame,
) -> np.ndarray:
    """Get prediction probabilities for patients.

    Args:
        model: Trained model.
        preprocessor: Fitted preprocessor.
        df: DataFrame with patient features.

    Returns:
        Array of probabilities for positive class.
    """
    X = preprocessor.transform(df)
    return model.predict_proba(X)[:, 1]
