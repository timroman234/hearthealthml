"""Data splitting utilities."""

from pathlib import Path

import joblib
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

from src.utils.logger import get_logger

logger = get_logger(__name__)


def create_splits(
    df: pd.DataFrame,
    target_col: str = "target",
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_state: int = 42,
) -> dict[str, pd.DataFrame | pd.Series]:
    """Create stratified train/val/test splits.

    Args:
        df: DataFrame to split.
        target_col: Name of target column.
        train_ratio: Proportion for training set.
        val_ratio: Proportion for validation set.
        test_ratio: Proportion for test set.
        random_state: Random seed for reproducibility.

    Returns:
        Dictionary with X_train, y_train, X_val, y_val, X_test, y_test.

    Raises:
        ValueError: If ratios don't sum to 1.0.
    """
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError(
            f"Ratios must sum to 1.0, got {train_ratio + val_ratio + test_ratio}"
        )

    X = df.drop(columns=[target_col])
    y = df[target_col]

    # First split: train vs (val + test)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=(val_ratio + test_ratio),
        stratify=y,
        random_state=random_state,
    )

    # Second split: val vs test
    val_ratio_adjusted = val_ratio / (val_ratio + test_ratio)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=(1 - val_ratio_adjusted),
        stratify=y_temp,
        random_state=random_state,
    )

    splits = {
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
        "X_test": X_test,
        "y_test": y_test,
    }

    logger.info(
        f"Created splits - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}"
    )

    # Log class distribution
    for name, y_split in [("train", y_train), ("val", y_val), ("test", y_test)]:
        pos_ratio = y_split.mean()
        logger.info(f"  {name} positive class ratio: {pos_ratio:.2%}")

    return splits


def create_cv_folds(
    n_folds: int = 5,
    shuffle: bool = True,
    random_state: int = 42,
) -> StratifiedKFold:
    """Create stratified K-Fold cross-validation iterator.

    Args:
        n_folds: Number of folds.
        shuffle: Whether to shuffle before splitting.
        random_state: Random seed for reproducibility.

    Returns:
        StratifiedKFold iterator.
    """
    cv = StratifiedKFold(n_splits=n_folds, shuffle=shuffle, random_state=random_state)
    logger.info(f"Created {n_folds}-fold stratified CV iterator")
    return cv


def save_splits(splits: dict, output_dir: Path) -> None:
    """Save splits to disk as pickle files.

    Args:
        splits: Dictionary of split DataFrames/Series.
        output_dir: Directory to save splits.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for name, data in splits.items():
        path = output_dir / f"{name}.pkl"
        joblib.dump(data, path)

    logger.info(f"Saved {len(splits)} splits to {output_dir}")


def load_splits(input_dir: Path) -> dict[str, pd.DataFrame | pd.Series]:
    """Load splits from disk.

    Args:
        input_dir: Directory containing split files.

    Returns:
        Dictionary of split DataFrames/Series.
    """
    input_dir = Path(input_dir)
    splits = {}

    for name in ["X_train", "y_train", "X_val", "y_val", "X_test", "y_test"]:
        path = input_dir / f"{name}.pkl"
        if path.exists():
            splits[name] = joblib.load(path)

    logger.info(f"Loaded {len(splits)} splits from {input_dir}")
    return splits
