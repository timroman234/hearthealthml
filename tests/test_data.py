"""Tests for data loading and validation."""

import sys
from pathlib import Path

import pandas as pd
import pytest

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.loader import EXPECTED_COLUMNS, get_data_info, load_raw_data, validate_schema
from src.data.preprocessor import (
    check_missing_values,
    create_preprocessor,
    detect_outliers,
    validate_ranges,
)
from src.data.splitter import create_splits


class TestDataLoader:
    """Tests for data loading functions."""

    @pytest.fixture
    def sample_df(self):
        """Create a sample DataFrame for testing."""
        return pd.DataFrame({
            "age": [55, 45, 65],
            "sex": [1, 0, 1],
            "cp": [0, 1, 2],
            "trestbps": [130, 120, 150],
            "chol": [250, 200, 300],
            "fbs": [0, 1, 0],
            "restecg": [0, 1, 0],
            "thalach": [150, 170, 130],
            "exang": [0, 1, 0],
            "oldpeak": [1.5, 0.5, 2.5],
            "slope": [1, 0, 2],
            "ca": [0, 1, 2],
            "thal": [2, 1, 3],
            "target": [1, 0, 1],
        })

    def test_load_raw_data_file_not_found(self):
        """Test that FileNotFoundError is raised for missing file."""
        with pytest.raises(FileNotFoundError):
            load_raw_data(Path("nonexistent.csv"))

    def test_validate_schema_valid(self, sample_df):
        """Test schema validation with valid DataFrame."""
        assert validate_schema(sample_df) is True

    def test_validate_schema_missing_columns(self, sample_df):
        """Test schema validation with missing columns."""
        df = sample_df.drop(columns=["age"])
        with pytest.raises(ValueError, match="Missing columns"):
            validate_schema(df)

    def test_get_data_info(self, sample_df):
        """Test data info extraction."""
        info = get_data_info(sample_df)
        assert info["n_rows"] == 3
        assert info["n_columns"] == 14
        assert "age" in info["columns"]


class TestPreprocessor:
    """Tests for preprocessing functions."""

    @pytest.fixture
    def sample_df(self):
        """Create a sample DataFrame for testing."""
        return pd.DataFrame({
            "age": [55, 45, 65, None],
            "sex": [1, 0, 1, 1],
            "cp": [0, 1, 2, 0],
            "trestbps": [130, 120, 150, 140],
            "chol": [250, 200, 300, 220],
            "fbs": [0, 1, 0, 0],
            "restecg": [0, 1, 0, 1],
            "thalach": [150, 170, 130, 160],
            "exang": [0, 1, 0, 0],
            "oldpeak": [1.5, 0.5, 2.5, 1.0],
            "slope": [1, 0, 2, 1],
            "ca": [0, 1, 2, 0],
            "thal": [2, 1, 3, 2],
            "target": [1, 0, 1, 0],
        })

    def test_check_missing_values(self, sample_df):
        """Test missing value detection."""
        missing = check_missing_values(sample_df)
        assert "age" in missing
        assert missing["age"] == 1

    def test_detect_outliers_iqr(self):
        """Test outlier detection using IQR method."""
        df = pd.DataFrame({
            "value": [1, 2, 3, 4, 5, 100],  # 100 is an outlier
        })
        outliers = detect_outliers(df, columns=["value"], method="iqr")
        assert outliers["value"].iloc[-1] is True  # Last value is outlier

    def test_validate_ranges_valid(self, sample_df):
        """Test range validation with valid data."""
        df = sample_df.dropna()
        errors = validate_ranges(df)
        assert len(errors) == 0

    def test_validate_ranges_invalid(self):
        """Test range validation with invalid data."""
        df = pd.DataFrame({
            "age": [-5],  # Invalid age
            "sex": [2],   # Invalid sex
        })
        errors = validate_ranges(df)
        assert len(errors) > 0

    def test_create_preprocessor(self):
        """Test preprocessor creation."""
        preprocessor = create_preprocessor()
        assert preprocessor is not None
        assert len(preprocessor.transformers) == 3  # continuous, binary, categorical


class TestSplitter:
    """Tests for data splitting functions."""

    @pytest.fixture
    def sample_df(self):
        """Create a sample DataFrame for testing."""
        return pd.DataFrame({
            "feature1": range(100),
            "feature2": range(100, 200),
            "target": [0] * 50 + [1] * 50,
        })

    def test_create_splits_ratios(self, sample_df):
        """Test that splits have correct sizes."""
        splits = create_splits(
            sample_df,
            target_col="target",
            train_ratio=0.7,
            val_ratio=0.15,
            test_ratio=0.15,
        )

        total = len(sample_df)
        assert len(splits["X_train"]) == int(total * 0.7)
        assert len(splits["X_val"]) == int(total * 0.15)
        assert len(splits["X_test"]) == int(total * 0.15)

    def test_create_splits_stratified(self, sample_df):
        """Test that splits preserve target distribution."""
        splits = create_splits(sample_df, target_col="target")

        original_ratio = sample_df["target"].mean()
        train_ratio = splits["y_train"].mean()

        # Ratio should be approximately preserved
        assert abs(original_ratio - train_ratio) < 0.1

    def test_create_splits_invalid_ratios(self, sample_df):
        """Test that invalid ratios raise error."""
        with pytest.raises(ValueError):
            create_splits(
                sample_df,
                target_col="target",
                train_ratio=0.5,
                val_ratio=0.3,
                test_ratio=0.3,  # Sum > 1
            )
