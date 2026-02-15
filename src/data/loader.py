"""Data loading utilities."""

from pathlib import Path

import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)

EXPECTED_COLUMNS = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
    "target",
]


def load_raw_data(path: Path) -> pd.DataFrame:
    """Load CSV data with error handling.

    Args:
        path: Path to the CSV file.

    Returns:
        DataFrame with loaded data.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file cannot be parsed as CSV.
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")

    logger.info(f"Loading data from {path}")
    df = pd.read_csv(path)
    logger.info(f"Loaded {len(df)} rows, {len(df.columns)} columns")

    return df


def validate_schema(
    df: pd.DataFrame, expected_columns: list[str] | None = None
) -> bool:
    """Validate DataFrame has expected columns.

    Args:
        df: DataFrame to validate.
        expected_columns: List of expected column names. Uses default if None.

    Returns:
        True if schema is valid.

    Raises:
        ValueError: If schema validation fails.
    """
    if expected_columns is None:
        expected_columns = EXPECTED_COLUMNS

    missing = set(expected_columns) - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    extra = set(df.columns) - set(expected_columns)
    if extra:
        logger.warning(f"Extra columns found (will be ignored): {extra}")

    logger.info("Schema validation passed")
    return True


def get_data_info(df: pd.DataFrame) -> dict:
    """Return metadata about the DataFrame.

    Args:
        df: DataFrame to analyze.

    Returns:
        Dictionary with shape, dtypes, and missing counts.
    """
    return {
        "n_rows": len(df),
        "n_columns": len(df.columns),
        "columns": list(df.columns),
        "dtypes": df.dtypes.astype(str).to_dict(),
        "missing_counts": df.isnull().sum().to_dict(),
        "memory_usage_mb": df.memory_usage(deep=True).sum() / 1024 / 1024,
    }
