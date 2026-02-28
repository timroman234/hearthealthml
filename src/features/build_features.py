"""Feature engineering transformations."""

import numpy as np
import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)

# Recommended interaction pairs
INTERACTION_PAIRS = [
    ("age", "thalach"),
    ("oldpeak", "slope"),
    ("cp", "exang"),
]


def create_age_groups(age: pd.Series) -> pd.Series:
    """Bin age into clinical categories.

    Args:
        age: Series of age values.

    Returns:
        Categorical series with age groups.
    """
    bins = [0, 40, 55, 70, 120]
    labels = ["young", "middle", "senior", "elderly"]
    return pd.cut(age, bins=bins, labels=labels)


def create_bp_category(trestbps: pd.Series) -> pd.Series:
    """Classify blood pressure per clinical guidelines.

    Args:
        trestbps: Series of resting blood pressure values.

    Returns:
        Series with BP categories.
    """
    conditions = [
        trestbps < 120,
        (trestbps >= 120) & (trestbps < 130),
        (trestbps >= 130) & (trestbps < 140),
        trestbps >= 140,
    ]
    categories = ["normal", "elevated", "high_stage1", "high_stage2"]
    return pd.Series(
        np.select(conditions, categories, default="unknown"),
        index=trestbps.index,
    )


def create_cholesterol_risk(chol: pd.Series) -> pd.Series:
    """Classify cholesterol risk levels.

    Args:
        chol: Series of cholesterol values.

    Returns:
        Series with cholesterol risk categories.
    """
    conditions = [
        chol < 200,
        (chol >= 200) & (chol < 240),
        chol >= 240,
    ]
    labels = ["desirable", "borderline", "high"]
    return pd.Series(
        np.select(conditions, labels, default="unknown"),
        index=chol.index,
    )


def calculate_heart_rate_reserve(age: pd.Series, thalach: pd.Series) -> pd.Series:
    """Calculate heart rate reserve.

    Heart rate reserve = max predicted HR (220 - age) - achieved HR

    Args:
        age: Series of age values.
        thalach: Series of maximum heart rate achieved.

    Returns:
        Series with heart rate reserve values.
    """
    max_predicted = 220 - age
    return max_predicted - thalach


def calculate_st_risk_score(oldpeak: pd.Series, slope: pd.Series) -> pd.Series:
    """Calculate ST segment risk score.

    Args:
        oldpeak: ST depression induced by exercise.
        slope: Slope of peak exercise ST segment.

    Returns:
        Series with ST risk scores.
    """
    return oldpeak * (slope + 1)


def calculate_cardiac_risk_score(df: pd.DataFrame) -> pd.Series:
    """Calculate composite cardiac risk score.

    Risk factors:
    - Age > 55: +1
    - Male: +1
    - Typical angina (cp=0): +2
    - High BP (trestbps >= 140): +1
    - High cholesterol (chol >= 240): +1
    - High fasting blood sugar: +1
    - Exercise induced angina: +2
    - High ST depression (oldpeak > 2): +2
    - Vessels colored > 0: +2

    Args:
        df: DataFrame with patient features.

    Returns:
        Series with cardiac risk scores (0-13).
    """
    score = (
        (df["age"] > 55).astype(int) * 1
        + (df["sex"] == 1).astype(int) * 1
        + (df["cp"] == 0).astype(int) * 2
        + (df["trestbps"] >= 140).astype(int) * 1
        + (df["chol"] >= 240).astype(int) * 1
        + (df["fbs"] == 1).astype(int) * 1
        + (df["exang"] == 1).astype(int) * 2
        + (df["oldpeak"] > 2).astype(int) * 2
        + (df["ca"] > 0).astype(int) * 2
    )
    return score


def create_interaction_features(
    df: pd.DataFrame, pairs: list[tuple[str, str]] | None = None
) -> pd.DataFrame:
    """Create interaction terms between feature pairs.

    Args:
        df: DataFrame with features.
        pairs: List of (feature1, feature2) tuples. Uses default if None.

    Returns:
        DataFrame with added interaction features.
    """
    if pairs is None:
        pairs = INTERACTION_PAIRS

    df = df.copy()
    for f1, f2 in pairs:
        if f1 in df.columns and f2 in df.columns:
            df[f"{f1}_x_{f2}"] = df[f1] * df[f2]
            logger.debug(f"Created interaction feature: {f1}_x_{f2}")

    return df


def engineer_features(
    df: pd.DataFrame,
    create_age_group: bool = True,
    create_bp_cat: bool = True,
    create_chol_risk: bool = True,
    create_hr_reserve: bool = True,
    create_risk_score: bool = True,
    create_interactions: bool = True,
) -> pd.DataFrame:
    """Apply all feature engineering transformations.

    Args:
        df: DataFrame with original features.
        create_age_group: Whether to create age group feature.
        create_bp_cat: Whether to create BP category feature.
        create_chol_risk: Whether to create cholesterol risk feature.
        create_hr_reserve: Whether to create heart rate reserve feature.
        create_risk_score: Whether to create cardiac risk score feature.
        create_interactions: Whether to create interaction features.

    Returns:
        DataFrame with engineered features added.
    """
    df = df.copy()
    n_original = len(df.columns)

    if create_age_group and "age" in df.columns:
        df["age_group"] = create_age_groups(df["age"])

    if create_bp_cat and "trestbps" in df.columns:
        df["bp_category"] = create_bp_category(df["trestbps"])

    if create_chol_risk and "chol" in df.columns:
        df["cholesterol_risk"] = create_cholesterol_risk(df["chol"])

    if create_hr_reserve and "age" in df.columns and "thalach" in df.columns:
        df["heart_rate_reserve"] = calculate_heart_rate_reserve(
            df["age"], df["thalach"]
        )

    if create_risk_score:
        required = [
            "age",
            "sex",
            "cp",
            "trestbps",
            "chol",
            "fbs",
            "exang",
            "oldpeak",
            "ca",
        ]
        if all(col in df.columns for col in required):
            df["cardiac_risk_score"] = calculate_cardiac_risk_score(df)

    if create_interactions:
        df = create_interaction_features(df)

    n_new = len(df.columns) - n_original
    logger.info(f"Created {n_new} new features, total: {len(df.columns)}")

    return df
