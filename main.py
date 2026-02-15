#!/usr/bin/env python
"""HeartHealthML - Full pipeline execution."""

import argparse
import json
from datetime import datetime
from pathlib import Path

from src.data.loader import load_raw_data, validate_schema
from src.data.preprocessor import (
    check_missing_values,
    create_preprocessor,
    detect_outliers,
    fit_transform_preprocessor,
    save_preprocessor,
    transform_preprocessor,
    validate_ranges,
)
from src.data.splitter import create_splits, save_splits
from src.evaluation.evaluate import (
    evaluate_model,
    find_optimal_threshold,
    plot_confusion_matrix,
    plot_roc_curve,
)
from src.features.build_features import engineer_features
from src.models.registry import ModelRegistry
from src.models.train import get_model, train_model, tune_hyperparameters
from src.utils.config import get_config
from src.utils.logger import get_logger, setup_logger


def main(config_path: str | None = None, tune: bool = False) -> dict:
    """Run the full ML pipeline.

    Args:
        config_path: Path to config file. Uses default if None.
        tune: Whether to perform hyperparameter tuning.

    Returns:
        Dictionary with pipeline results.
    """
    # Setup logging
    logger = setup_logger("hearthealthml", log_file=Path("logs/pipeline.log"))
    logger.info("=" * 60)
    logger.info("Starting HeartHealthML Pipeline")
    logger.info("=" * 60)

    # Load config
    if config_path:
        from src.utils.config import load_config
        config = load_config(Path(config_path))
    else:
        config = get_config("config")

    # Stage 1: Data Loading
    logger.info("Stage 1: Loading data")
    raw_path = Path(config["data"]["raw_path"])
    df = load_raw_data(raw_path)
    validate_schema(df)

    # Stage 2: Data Validation
    logger.info("Stage 2: Validating data")
    missing = check_missing_values(df)
    if missing:
        logger.warning(f"Found missing values: {missing}")
        # For now, drop rows with missing values
        df = df.dropna()
        logger.info(f"Dropped rows with missing values. New shape: {df.shape}")

    outliers = detect_outliers(df)
    validation_errors = validate_ranges(df)
    if validation_errors:
        logger.warning(f"Validation errors: {validation_errors}")

    # Stage 3: Feature Engineering
    logger.info("Stage 3: Engineering features")
    fe_config = config["features"].get("engineering", {})
    if fe_config.get("enabled", True):
        df = engineer_features(
            df,
            create_age_group=fe_config.get("create_age_groups", True),
            create_bp_cat=fe_config.get("create_bp_category", True),
            create_chol_risk=fe_config.get("create_cholesterol_risk", True),
            create_hr_reserve=fe_config.get("create_heart_rate_reserve", True),
            create_risk_score=fe_config.get("create_cardiac_risk_score", True),
            create_interactions=True,
        )

    # Stage 4: Data Splitting
    logger.info("Stage 4: Splitting data")
    target_col = config["features"]["target"]
    splits = create_splits(
        df,
        target_col=target_col,
        train_ratio=config["splitting"]["train_ratio"],
        val_ratio=config["splitting"]["val_ratio"],
        test_ratio=config["splitting"]["test_ratio"],
        random_state=config["splitting"]["random_state"],
    )

    # Save splits
    splits_path = Path(config["data"]["splits_path"])
    save_splits(splits, splits_path)

    # Stage 5: Preprocessing
    logger.info("Stage 5: Preprocessing features")

    # Get feature lists (excluding engineered categorical features for simplicity)
    continuous = config["features"]["continuous"]
    binary = config["features"]["binary"]
    categorical = config["features"]["categorical"]

    # Add numeric engineered features to continuous
    engineered_numeric = ["heart_rate_reserve", "cardiac_risk_score"]
    for feat in engineered_numeric:
        if feat in splits["X_train"].columns:
            continuous = continuous + [feat]

    # Add interaction features to continuous
    for col in splits["X_train"].columns:
        if "_x_" in col and col not in continuous:
            continuous = continuous + [col]

    preprocessor = create_preprocessor(
        continuous_features=[c for c in continuous if c in splits["X_train"].columns],
        binary_features=[b for b in binary if b in splits["X_train"].columns],
        categorical_features=[c for c in categorical if c in splits["X_train"].columns],
        scaler=config["preprocessing"]["scaler"],
    )

    X_train = fit_transform_preprocessor(preprocessor, splits["X_train"])
    X_val = transform_preprocessor(preprocessor, splits["X_val"])
    X_test = transform_preprocessor(preprocessor, splits["X_test"])

    y_train = splits["y_train"].values
    y_val = splits["y_val"].values
    y_test = splits["y_test"].values

    # Save preprocessor
    processed_path = Path(config["data"]["processed_path"])
    processed_path.mkdir(parents=True, exist_ok=True)
    save_preprocessor(preprocessor, processed_path / "preprocessor.joblib")

    # Stage 6: Model Training
    logger.info("Stage 6: Training model")
    model_name = config["training"]["default_model"]

    if tune:
        # Stage 7: Hyperparameter Tuning
        logger.info("Stage 7: Tuning hyperparameters")
        tune_results = tune_hyperparameters(
            model_name,
            X_train,
            y_train,
            cv=config["training"]["cv_folds"],
            scoring=config["evaluation"]["primary_metric"],
        )
        model = tune_results["best_estimator"]
        best_params = tune_results["best_params"]
        logger.info(f"Best params: {best_params}")
    else:
        model = get_model(model_name)
        model = train_model(model, X_train, y_train, X_val, y_val)
        best_params = None

    # Stage 8: Find Optimal Threshold
    logger.info("Stage 8: Finding optimal threshold")
    if config["evaluation"].get("optimize_threshold", False):
        threshold = find_optimal_threshold(
            model, X_val, y_val,
            optimize_for=config["evaluation"].get("threshold_metric", "f1")
        )
    else:
        threshold = config["evaluation"]["threshold"]

    # Stage 9: Model Evaluation
    logger.info("Stage 9: Evaluating model")
    metrics = evaluate_model(model, X_test, y_test, threshold=threshold)

    # Save evaluation plots
    figures_dir = Path(config["output"]["figures_dir"])
    figures_dir.mkdir(parents=True, exist_ok=True)

    y_pred = (model.predict_proba(X_test)[:, 1] >= threshold).astype(int)
    plot_confusion_matrix(y_test, y_pred, figures_dir / "confusion_matrix.png")
    plot_roc_curve(model, X_test, y_test, figures_dir / "roc_curve.png")

    # Save metrics
    metrics_dir = Path(config["output"]["metrics_dir"])
    metrics_dir.mkdir(parents=True, exist_ok=True)
    with open(metrics_dir / "evaluation_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2, default=str)

    # Stage 10: Register Model
    logger.info("Stage 10: Registering model")
    registry = ModelRegistry(Path(config["output"]["models_dir"]))

    metadata = {
        "model_name": model_name,
        "features_used": list(splits["X_train"].columns),
        "feature_engineering": fe_config.get("enabled", True),
        "training_samples": len(X_train),
        "validation_samples": len(X_val),
        "test_samples": len(X_test),
        "threshold": threshold,
        "tuned": tune,
        "best_params": best_params,
    }

    version = registry.register_model(model, preprocessor, metrics, metadata)

    # Summary
    logger.info("=" * 60)
    logger.info("Pipeline completed successfully!")
    logger.info(f"Model: {model_name} v{version}")
    logger.info(f"Test Accuracy: {metrics['accuracy']:.4f}")
    logger.info(f"Test ROC-AUC: {metrics.get('roc_auc', 'N/A'):.4f}")
    logger.info(f"Optimal Threshold: {threshold:.2f}")
    logger.info("=" * 60)

    return {
        "model_name": model_name,
        "version": version,
        "metrics": metrics,
        "threshold": threshold,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run HeartHealthML Pipeline")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config file",
    )
    parser.add_argument(
        "--tune",
        action="store_true",
        help="Perform hyperparameter tuning",
    )

    args = parser.parse_args()
    results = main(config_path=args.config, tune=args.tune)

    print(f"\nPipeline Results:")
    print(f"  Model: {results['model_name']} v{results['version']}")
    print(f"  Test Accuracy: {results['metrics']['accuracy']:.4f}")
    print(f"  Test ROC-AUC: {results['metrics'].get('roc_auc', 'N/A'):.4f}")
