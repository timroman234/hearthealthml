"""Model training logic."""

import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from src.utils.logger import get_logger

logger = get_logger(__name__)

# Model registry
MODEL_REGISTRY = {
    "logistic_regression": LogisticRegression,
}

# Default hyperparameters
DEFAULT_PARAMS = {
    "logistic_regression": {
        "C": 1.0,
        "penalty": "l2",
        "solver": "liblinear",
        "max_iter": 200,
        "random_state": 42,
    },
}

# Hyperparameter search spaces
PARAM_GRIDS = {
    "logistic_regression": {
        "C": [0.001, 0.01, 0.1, 1, 10, 100],
        "penalty": ["l1", "l2"],
        "solver": ["liblinear"],
        "max_iter": [100, 200, 500],
        "class_weight": [None, "balanced"],
    },
}


def get_model(
    model_name: str, params: dict | None = None
) -> BaseEstimator:
    """Factory function to create model instances.

    Args:
        model_name: Name of the model to create.
        params: Model hyperparameters. Uses defaults if None.

    Returns:
        Configured model instance.

    Raises:
        ValueError: If model_name is not registered.
    """
    if model_name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model: {model_name}. Available: {list(MODEL_REGISTRY.keys())}"
        )

    if params is None:
        params = DEFAULT_PARAMS.get(model_name, {})

    model_class = MODEL_REGISTRY[model_name]
    model = model_class(**params)

    logger.info(f"Created {model_name} with params: {params}")
    return model


def train_model(
    model: BaseEstimator,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray | None = None,
    y_val: np.ndarray | None = None,
) -> BaseEstimator:
    """Train model and optionally evaluate on validation set.

    Args:
        model: Model to train.
        X_train: Training features.
        y_train: Training labels.
        X_val: Validation features (optional).
        y_val: Validation labels (optional).

    Returns:
        Trained model.
    """
    logger.info(f"Training model on {len(X_train)} samples")
    model.fit(X_train, y_train)

    train_score = model.score(X_train, y_train)
    logger.info(f"Training accuracy: {train_score:.4f}")

    if X_val is not None and y_val is not None:
        val_score = model.score(X_val, y_val)
        logger.info(f"Validation accuracy: {val_score:.4f}")

    return model


def tune_hyperparameters(
    model_name: str,
    X: np.ndarray,
    y: np.ndarray,
    param_grid: dict | None = None,
    cv: int = 5,
    scoring: str = "roc_auc",
    n_jobs: int = -1,
) -> dict:
    """Perform grid search with cross-validation.

    Args:
        model_name: Name of the model to tune.
        X: Feature matrix.
        y: Target vector.
        param_grid: Hyperparameter grid. Uses defaults if None.
        cv: Number of CV folds.
        scoring: Scoring metric.
        n_jobs: Number of parallel jobs.

    Returns:
        Dictionary with best_params, best_score, best_estimator, cv_results.
    """
    if param_grid is None:
        param_grid = PARAM_GRIDS.get(model_name, {})

    base_model = get_model(model_name, {})

    cv_splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)

    logger.info(f"Starting grid search for {model_name} with {cv}-fold CV")
    logger.info(f"Parameter grid: {param_grid}")

    grid_search = GridSearchCV(
        base_model,
        param_grid,
        cv=cv_splitter,
        scoring=scoring,
        n_jobs=n_jobs,
        verbose=1,
        refit=True,
    )
    grid_search.fit(X, y)

    logger.info(f"Best {scoring}: {grid_search.best_score_:.4f}")
    logger.info(f"Best params: {grid_search.best_params_}")

    return {
        "best_params": grid_search.best_params_,
        "best_score": grid_search.best_score_,
        "best_estimator": grid_search.best_estimator_,
        "cv_results": grid_search.cv_results_,
    }


def save_model(
    model: BaseEstimator,
    metadata: dict,
    output_path: Path,
) -> None:
    """Save model artifact with metadata.

    Args:
        model: Trained model to save.
        metadata: Training metadata.
        output_path: Directory to save model.
    """
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save model
    model_file = output_path / "model.joblib"
    joblib.dump(model, model_file)
    logger.info(f"Saved model to {model_file}")

    # Save metadata
    metadata_file = output_path / "metadata.json"
    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=2, default=str)
    logger.info(f"Saved metadata to {metadata_file}")


def load_model(model_path: Path) -> BaseEstimator:
    """Load model from disk.

    Args:
        model_path: Path to model directory or model.joblib file.

    Returns:
        Loaded model.
    """
    model_path = Path(model_path)

    if model_path.is_dir():
        model_path = model_path / "model.joblib"

    model = joblib.load(model_path)
    logger.info(f"Loaded model from {model_path}")
    return model
