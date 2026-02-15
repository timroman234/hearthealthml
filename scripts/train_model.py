#!/usr/bin/env python
"""Model training script."""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.loader import load_raw_data
from src.data.preprocessor import create_preprocessor, fit_transform_preprocessor
from src.data.splitter import create_splits
from src.evaluation.evaluate import evaluate_model
from src.features.build_features import engineer_features
from src.models.train import get_model, train_model, tune_hyperparameters
from src.utils.config import get_config
from src.utils.logger import setup_logger


def main(
    model_name: str = "logistic_regression",
    tune: bool = False,
) -> None:
    """Train a model.

    Args:
        model_name: Name of the model to train.
        tune: Whether to perform hyperparameter tuning.
    """
    logger = setup_logger("hearthealthml")
    config = get_config("config")

    logger.info(f"Training {model_name}")

    # Load and prepare data
    df = load_raw_data(Path(config["data"]["raw_path"]))
    df = engineer_features(df)

    # Split data
    splits = create_splits(
        df,
        target_col=config["features"]["target"],
        train_ratio=config["splitting"]["train_ratio"],
        val_ratio=config["splitting"]["val_ratio"],
        test_ratio=config["splitting"]["test_ratio"],
    )

    # Preprocess
    preprocessor = create_preprocessor()
    X_train = fit_transform_preprocessor(preprocessor, splits["X_train"])
    X_test = preprocessor.transform(splits["X_test"])

    y_train = splits["y_train"].values
    y_test = splits["y_test"].values

    # Train
    if tune:
        results = tune_hyperparameters(model_name, X_train, y_train)
        model = results["best_estimator"]
        logger.info(f"Best params: {results['best_params']}")
    else:
        model = get_model(model_name)
        model = train_model(model, X_train, y_train)

    # Evaluate
    metrics = evaluate_model(model, X_test, y_test)

    print(f"\nResults for {model_name}:")
    print(f"  Accuracy: {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall: {metrics['recall']:.4f}")
    print(f"  F1: {metrics['f1']:.4f}")
    print(f"  ROC-AUC: {metrics.get('roc_auc', 'N/A'):.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a model")
    parser.add_argument(
        "--model",
        type=str,
        default="logistic_regression",
        help="Model name to train",
    )
    parser.add_argument(
        "--tune",
        action="store_true",
        help="Perform hyperparameter tuning",
    )

    args = parser.parse_args()
    main(model_name=args.model, tune=args.tune)
