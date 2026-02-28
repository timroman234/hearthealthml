"""Tests for feature engineering."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.features.build_features import (
    calculate_cardiac_risk_score,
    calculate_heart_rate_reserve,
    create_age_groups,
    create_bp_category,
    create_cholesterol_risk,
    create_interaction_features,
    engineer_features,
)
from src.features.selection import (
    get_feature_importances,
    remove_correlated_features,
    select_by_variance,
)


class TestFeatureEngineering:
    """Tests for feature engineering functions."""

    def test_create_age_groups(self):
        """Test age group binning."""
        ages = pd.Series([25, 45, 60, 75])
        groups = create_age_groups(ages)

        assert groups.iloc[0] == "young"
        assert groups.iloc[1] == "middle"
        assert groups.iloc[2] == "senior"
        assert groups.iloc[3] == "elderly"

    def test_create_bp_category(self):
        """Test blood pressure categorization."""
        bp = pd.Series([110, 125, 135, 150])
        categories = create_bp_category(bp)

        assert categories.iloc[0] == "normal"
        assert categories.iloc[1] == "elevated"
        assert categories.iloc[2] == "high_stage1"
        assert categories.iloc[3] == "high_stage2"

    def test_create_cholesterol_risk(self):
        """Test cholesterol risk categorization."""
        chol = pd.Series([180, 220, 260])
        risk = create_cholesterol_risk(chol)

        assert risk.iloc[0] == "desirable"
        assert risk.iloc[1] == "borderline"
        assert risk.iloc[2] == "high"

    def test_calculate_heart_rate_reserve(self):
        """Test heart rate reserve calculation."""
        age = pd.Series([40, 50, 60])
        thalach = pd.Series([150, 140, 130])

        reserve = calculate_heart_rate_reserve(age, thalach)

        # 220 - 40 - 150 = 30
        assert reserve.iloc[0] == 30
        # 220 - 50 - 140 = 30
        assert reserve.iloc[1] == 30
        # 220 - 60 - 130 = 30
        assert reserve.iloc[2] == 30

    def test_calculate_cardiac_risk_score_range(self):
        """Test that cardiac risk score is within expected range."""
        df = pd.DataFrame(
            {
                "age": [55, 30, 70],
                "sex": [1, 0, 1],
                "cp": [0, 3, 0],
                "trestbps": [145, 110, 160],
                "chol": [250, 180, 280],
                "fbs": [1, 0, 1],
                "exang": [1, 0, 1],
                "oldpeak": [3.0, 0.5, 4.0],
                "ca": [2, 0, 3],
            }
        )

        scores = calculate_cardiac_risk_score(df)

        # Score should be between 0 and 13
        assert scores.min() >= 0
        assert scores.max() <= 13

    def test_create_interaction_features(self):
        """Test interaction feature creation."""
        df = pd.DataFrame(
            {
                "age": [50, 60],
                "thalach": [150, 140],
                "oldpeak": [1.0, 2.0],
                "slope": [1, 2],
            }
        )

        pairs = [("age", "thalach"), ("oldpeak", "slope")]
        result = create_interaction_features(df, pairs)

        assert "age_x_thalach" in result.columns
        assert "oldpeak_x_slope" in result.columns
        assert result["age_x_thalach"].iloc[0] == 50 * 150
        assert result["oldpeak_x_slope"].iloc[0] == 1.0 * 1

    def test_engineer_features(self):
        """Test full feature engineering."""
        df = pd.DataFrame(
            {
                "age": [55, 45],
                "sex": [1, 0],
                "cp": [0, 1],
                "trestbps": [140, 120],
                "chol": [250, 190],
                "fbs": [1, 0],
                "restecg": [0, 1],
                "thalach": [150, 170],
                "exang": [1, 0],
                "oldpeak": [2.0, 0.5],
                "slope": [1, 0],
                "ca": [1, 0],
                "thal": [2, 1],
            }
        )

        result = engineer_features(df)

        # Check that new features were created
        assert "age_group" in result.columns
        assert "bp_category" in result.columns
        assert "cholesterol_risk" in result.columns
        assert "heart_rate_reserve" in result.columns
        assert "cardiac_risk_score" in result.columns


class TestFeatureSelection:
    """Tests for feature selection functions."""

    def test_remove_correlated_features(self):
        """Test highly correlated feature removal."""
        df = pd.DataFrame(
            {
                "a": [1, 2, 3, 4, 5],
                "b": [1.01, 2.01, 3.01, 4.01, 5.01],  # Highly correlated with a
                "c": [5, 4, 3, 2, 1],  # Negatively correlated
            }
        )

        to_drop = remove_correlated_features(df, threshold=0.99)
        assert "b" in to_drop or "a" in to_drop

    def test_select_by_variance(self):
        """Test low variance feature removal."""
        df = pd.DataFrame(
            {
                "constant": [1, 1, 1, 1, 1],
                "variable": [1, 2, 3, 4, 5],
            }
        )

        selected = select_by_variance(df, threshold=0.01)
        assert "variable" in selected
        assert "constant" not in selected

    def test_get_feature_importances(self):
        """Test feature importance extraction."""
        from sklearn.linear_model import LogisticRegression

        X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
        y = np.array([0, 0, 1, 1])

        model = LogisticRegression()
        model.fit(X, y)

        importances = get_feature_importances(model, ["feature1", "feature2"])

        assert len(importances) == 2
        assert "feature" in importances.columns
        assert "importance" in importances.columns
