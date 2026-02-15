"""Data preprocessing and validation."""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.utils.logger import get_logger

logger = get_logger(__name__)

# Feature categories
CONTINUOUS_FEATURES = ["age", "trestbps", "chol", "thalach", "oldpeak"]
BINARY_FEATURES = ["sex", "fbs", "exang"]
CATEGORICAL_FEATURES = ["cp", "restecg", "slope", "ca", "thal"]

# Validation rules
VALIDATION_RULES = {
    "age": {"min": 0, "max": 120},
    "sex": {"values": [0, 1]},
    "cp": {"values": [0, 1, 2, 3]},
    "trestbps": {"min": 0, "max": 300},
    "chol": {"min": 0, "max": 600},
    "fbs": {"values": [0, 1]},
    "restecg": {"values": [0, 1, 2]},
    "thalach": {"min": 0, "max": 250},
    "exang": {"values": [0, 1]},
    "oldpeak": {"min": 0, "max": 10},
    "slope": {"values": [0, 1, 2]},
    "ca": {"values": [0, 1, 2, 3, 4]},
    "thal": {"values": [0, 1, 2, 3]},
    "target": {"values": [0, 1]},
}


def check_missing_values(df: pd.DataFrame) -> dict[str, int]:
    """Return count of missing values per column.

    Args:
        df: DataFrame to check.

    Returns:
        Dictionary mapping column names to missing counts.
    """
    missing = df.isnull().sum()
    missing_dict = missing[missing > 0].to_dict()

    if missing_dict:
        logger.warning(f"Missing values found: {missing_dict}")
    else:
        logger.info("No missing values found")

    return missing_dict


def detect_outliers(
    df: pd.DataFrame,
    columns: list[str] | None = None,
    method: str = "iqr",
    threshold: float = 1.5,
) -> pd.DataFrame:
    """Detect outliers using IQR or Z-score method.

    Args:
        df: DataFrame to analyze.
        columns: Columns to check. Uses continuous features if None.
        method: Detection method ('iqr' or 'zscore').
        threshold: IQR multiplier or Z-score threshold.

    Returns:
        DataFrame with boolean mask indicating outliers.
    """
    if columns is None:
        columns = [c for c in CONTINUOUS_FEATURES if c in df.columns]

    outlier_mask = pd.DataFrame(False, index=df.index, columns=columns)

    for col in columns:
        if col not in df.columns:
            continue

        if method == "iqr":
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            lower = q1 - threshold * iqr
            upper = q3 + threshold * iqr
            outlier_mask[col] = (df[col] < lower) | (df[col] > upper)
        elif method == "zscore":
            z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
            outlier_mask[col] = z_scores > threshold

    n_outliers = outlier_mask.any(axis=1).sum()
    logger.info(f"Detected {n_outliers} rows with outliers using {method} method")

    return outlier_mask


def validate_ranges(
    df: pd.DataFrame, rules: dict | None = None
) -> list[str]:
    """Validate values are within expected ranges.

    Args:
        df: DataFrame to validate.
        rules: Validation rules. Uses default if None.

    Returns:
        List of validation error messages.
    """
    if rules is None:
        rules = VALIDATION_RULES

    errors = []

    for col, rule in rules.items():
        if col not in df.columns:
            continue

        if "values" in rule:
            invalid = ~df[col].isin(rule["values"])
            if invalid.any():
                bad_values = df.loc[invalid, col].unique().tolist()
                errors.append(
                    f"{col}: invalid values {bad_values}, expected {rule['values']}"
                )

        if "min" in rule:
            below_min = df[col] < rule["min"]
            if below_min.any():
                errors.append(
                    f"{col}: {below_min.sum()} values below minimum {rule['min']}"
                )

        if "max" in rule:
            above_max = df[col] > rule["max"]
            if above_max.any():
                errors.append(
                    f"{col}: {above_max.sum()} values above maximum {rule['max']}"
                )

    if errors:
        for error in errors:
            logger.warning(f"Validation error: {error}")
    else:
        logger.info("All values within expected ranges")

    return errors


def create_preprocessor(
    continuous_features: list[str] | None = None,
    binary_features: list[str] | None = None,
    categorical_features: list[str] | None = None,
    scaler: str = "standard",
) -> ColumnTransformer:
    """Create sklearn ColumnTransformer for preprocessing.

    Args:
        continuous_features: List of continuous feature names.
        binary_features: List of binary feature names.
        categorical_features: List of categorical feature names.
        scaler: Type of scaler ('standard', 'robust', 'minmax').

    Returns:
        Configured ColumnTransformer.
    """
    if continuous_features is None:
        continuous_features = CONTINUOUS_FEATURES
    if binary_features is None:
        binary_features = BINARY_FEATURES
    if categorical_features is None:
        categorical_features = CATEGORICAL_FEATURES

    # Select scaler
    if scaler == "standard":
        scaler_obj = StandardScaler()
    elif scaler == "robust":
        from sklearn.preprocessing import RobustScaler

        scaler_obj = RobustScaler()
    elif scaler == "minmax":
        from sklearn.preprocessing import MinMaxScaler

        scaler_obj = MinMaxScaler()
    else:
        raise ValueError(f"Unknown scaler: {scaler}")

    continuous_transformer = Pipeline([("scaler", scaler_obj)])

    categorical_transformer = Pipeline(
        [("encoder", OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore"))]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("continuous", continuous_transformer, continuous_features),
            ("binary", "passthrough", binary_features),
            ("categorical", categorical_transformer, categorical_features),
        ],
        remainder="drop",
    )

    logger.info(
        f"Created preprocessor with {len(continuous_features)} continuous, "
        f"{len(binary_features)} binary, {len(categorical_features)} categorical features"
    )

    return preprocessor


def fit_transform_preprocessor(
    preprocessor: ColumnTransformer, df: pd.DataFrame
) -> np.ndarray:
    """Fit preprocessor and transform data.

    Args:
        preprocessor: ColumnTransformer to fit.
        df: DataFrame to transform.

    Returns:
        Transformed numpy array.
    """
    X = preprocessor.fit_transform(df)
    logger.info(f"Transformed data shape: {X.shape}")
    return X


def transform_preprocessor(
    preprocessor: ColumnTransformer, df: pd.DataFrame
) -> np.ndarray:
    """Transform data using fitted preprocessor.

    Args:
        preprocessor: Fitted ColumnTransformer.
        df: DataFrame to transform.

    Returns:
        Transformed numpy array.
    """
    return preprocessor.transform(df)


def save_preprocessor(preprocessor: ColumnTransformer, path: Path) -> None:
    """Save fitted preprocessor to disk.

    Args:
        preprocessor: Fitted ColumnTransformer.
        path: Output path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(preprocessor, path)
    logger.info(f"Saved preprocessor to {path}")


def load_preprocessor(path: Path) -> ColumnTransformer:
    """Load preprocessor from disk.

    Args:
        path: Path to saved preprocessor.

    Returns:
        Loaded ColumnTransformer.
    """
    preprocessor = joblib.load(path)
    logger.info(f"Loaded preprocessor from {path}")
    return preprocessor
