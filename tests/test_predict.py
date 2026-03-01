"""Tests for prediction/inference functions."""

import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.predict import predict_batch, predict_proba, predict_single


class TestPredictSingle:
    """Tests for single patient prediction."""

    @pytest.fixture
    def model_and_preprocessor(self):
        """Create a simple model and preprocessor for testing."""
        # Create simple training data
        np.random.seed(42)
        X_train = np.random.randn(100, 3)
        y_train = (X_train[:, 0] + X_train[:, 1] > 0).astype(int)

        # Train model
        model = LogisticRegression(random_state=42)
        model.fit(X_train, y_train)

        # Create simple preprocessor
        preprocessor = ColumnTransformer(
            transformers=[("scaler", StandardScaler(), [0, 1, 2])],
            remainder="drop",
        )
        # Fit on DataFrame with named columns
        df_train = pd.DataFrame(X_train, columns=["feat1", "feat2", "feat3"])
        preprocessor.fit(df_train)

        return model, preprocessor

    def test_predict_single_returns_dict(self, model_and_preprocessor):
        """Test that predict_single returns a dictionary."""
        model, preprocessor = model_and_preprocessor
        patient_data = {"feat1": 0.5, "feat2": 0.3, "feat3": -0.2}

        result = predict_single(model, preprocessor, patient_data)

        assert isinstance(result, dict)

    def test_predict_single_has_required_keys(self, model_and_preprocessor):
        """Test that result has all required keys."""
        model, preprocessor = model_and_preprocessor
        patient_data = {"feat1": 0.5, "feat2": 0.3, "feat3": -0.2}

        result = predict_single(model, preprocessor, patient_data)

        assert "prediction" in result
        assert "prediction_label" in result
        assert "probability" in result
        assert "risk_level" in result
        assert "threshold_used" in result

    def test_predict_single_binary_prediction(self, model_and_preprocessor):
        """Test that prediction is binary."""
        model, preprocessor = model_and_preprocessor
        patient_data = {"feat1": 0.5, "feat2": 0.3, "feat3": -0.2}

        result = predict_single(model, preprocessor, patient_data)

        assert result["prediction"] in [0, 1]

    def test_predict_single_valid_probability(self, model_and_preprocessor):
        """Test that probability is valid."""
        model, preprocessor = model_and_preprocessor
        patient_data = {"feat1": 0.5, "feat2": 0.3, "feat3": -0.2}

        result = predict_single(model, preprocessor, patient_data)

        assert 0.0 <= result["probability"] <= 1.0

    def test_predict_single_risk_levels(self, model_and_preprocessor):
        """Test risk level assignment."""
        model, preprocessor = model_and_preprocessor

        # Test different probability ranges by checking risk level is valid
        patient_data = {"feat1": 0.5, "feat2": 0.3, "feat3": -0.2}
        result = predict_single(model, preprocessor, patient_data)

        assert result["risk_level"] in ["low", "medium", "high"]

    def test_predict_single_custom_threshold(self, model_and_preprocessor):
        """Test prediction with custom threshold."""
        model, preprocessor = model_and_preprocessor
        patient_data = {"feat1": 0.5, "feat2": 0.3, "feat3": -0.2}

        result = predict_single(model, preprocessor, patient_data, threshold=0.3)

        assert result["threshold_used"] == 0.3

    def test_predict_single_label_matches_prediction(self, model_and_preprocessor):
        """Test that label matches prediction value."""
        model, preprocessor = model_and_preprocessor
        patient_data = {"feat1": 0.5, "feat2": 0.3, "feat3": -0.2}

        result = predict_single(model, preprocessor, patient_data)

        if result["prediction"] == 1:
            assert result["prediction_label"] == "heart_disease"
        else:
            assert result["prediction_label"] == "healthy"


class TestPredictBatch:
    """Tests for batch prediction."""

    @pytest.fixture
    def model_and_preprocessor(self):
        """Create a simple model and preprocessor for testing."""
        np.random.seed(42)
        X_train = np.random.randn(100, 3)
        y_train = (X_train[:, 0] + X_train[:, 1] > 0).astype(int)

        model = LogisticRegression(random_state=42)
        model.fit(X_train, y_train)

        preprocessor = ColumnTransformer(
            transformers=[("scaler", StandardScaler(), [0, 1, 2])],
            remainder="drop",
        )
        df_train = pd.DataFrame(X_train, columns=["feat1", "feat2", "feat3"])
        preprocessor.fit(df_train)

        return model, preprocessor

    def test_predict_batch_returns_dataframe(self, model_and_preprocessor):
        """Test that predict_batch returns a DataFrame."""
        model, preprocessor = model_and_preprocessor
        df = pd.DataFrame(
            {
                "feat1": [0.5, -0.5, 0.2],
                "feat2": [0.3, 0.8, -0.1],
                "feat3": [-0.2, 0.1, 0.5],
            }
        )

        result = predict_batch(model, preprocessor, df)

        assert isinstance(result, pd.DataFrame)

    def test_predict_batch_adds_columns(self, model_and_preprocessor):
        """Test that batch adds prediction columns."""
        model, preprocessor = model_and_preprocessor
        df = pd.DataFrame(
            {"feat1": [0.5, -0.5], "feat2": [0.3, 0.8], "feat3": [-0.2, 0.1]}
        )

        result = predict_batch(model, preprocessor, df)

        assert "prediction" in result.columns
        assert "probability" in result.columns
        assert "prediction_label" in result.columns
        assert "risk_level" in result.columns

    def test_predict_batch_preserves_original_data(self, model_and_preprocessor):
        """Test that original columns are preserved."""
        model, preprocessor = model_and_preprocessor
        df = pd.DataFrame(
            {"feat1": [0.5, -0.5], "feat2": [0.3, 0.8], "feat3": [-0.2, 0.1]}
        )

        result = predict_batch(model, preprocessor, df)

        assert "feat1" in result.columns
        assert "feat2" in result.columns
        assert "feat3" in result.columns

    def test_predict_batch_same_length(self, model_and_preprocessor):
        """Test that result has same number of rows."""
        model, preprocessor = model_and_preprocessor
        df = pd.DataFrame(
            {
                "feat1": [0.5, -0.5, 0.2],
                "feat2": [0.3, 0.8, -0.1],
                "feat3": [-0.2, 0.1, 0.5],
            }
        )

        result = predict_batch(model, preprocessor, df)

        assert len(result) == len(df)

    def test_predict_batch_custom_threshold(self, model_and_preprocessor):
        """Test batch prediction with custom threshold."""
        model, preprocessor = model_and_preprocessor
        df = pd.DataFrame(
            {"feat1": [0.5, -0.5], "feat2": [0.3, 0.8], "feat3": [-0.2, 0.1]}
        )

        result = predict_batch(model, preprocessor, df, threshold=0.3)

        # Should have predictions
        assert "prediction" in result.columns


class TestPredictProba:
    """Tests for probability prediction."""

    @pytest.fixture
    def model_and_preprocessor(self):
        """Create a simple model and preprocessor for testing."""
        np.random.seed(42)
        X_train = np.random.randn(100, 3)
        y_train = (X_train[:, 0] + X_train[:, 1] > 0).astype(int)

        model = LogisticRegression(random_state=42)
        model.fit(X_train, y_train)

        preprocessor = ColumnTransformer(
            transformers=[("scaler", StandardScaler(), [0, 1, 2])],
            remainder="drop",
        )
        df_train = pd.DataFrame(X_train, columns=["feat1", "feat2", "feat3"])
        preprocessor.fit(df_train)

        return model, preprocessor

    def test_predict_proba_returns_array(self, model_and_preprocessor):
        """Test that predict_proba returns numpy array."""
        model, preprocessor = model_and_preprocessor
        df = pd.DataFrame(
            {"feat1": [0.5, -0.5], "feat2": [0.3, 0.8], "feat3": [-0.2, 0.1]}
        )

        result = predict_proba(model, preprocessor, df)

        assert isinstance(result, np.ndarray)

    def test_predict_proba_correct_length(self, model_and_preprocessor):
        """Test that result has correct length."""
        model, preprocessor = model_and_preprocessor
        df = pd.DataFrame(
            {
                "feat1": [0.5, -0.5, 0.2],
                "feat2": [0.3, 0.8, -0.1],
                "feat3": [-0.2, 0.1, 0.5],
            }
        )

        result = predict_proba(model, preprocessor, df)

        assert len(result) == 3

    def test_predict_proba_valid_range(self, model_and_preprocessor):
        """Test that probabilities are in valid range."""
        model, preprocessor = model_and_preprocessor
        df = pd.DataFrame(
            {"feat1": [0.5, -0.5], "feat2": [0.3, 0.8], "feat3": [-0.2, 0.1]}
        )

        result = predict_proba(model, preprocessor, df)

        assert (result >= 0.0).all()
        assert (result <= 1.0).all()


class TestLoadProductionModel:
    """Tests for loading production models."""

    def test_load_production_model_not_found(self):
        """Test error when model doesn't exist."""
        from src.models.predict import load_production_model

        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(ValueError, match="Model not found"):
                load_production_model(
                    model_name="nonexistent",
                    registry_path=Path(tmpdir),
                )
