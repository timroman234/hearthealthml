"""Tests for model training and prediction."""

import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
from sklearn.datasets import make_classification

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.train import get_model, load_model, save_model, train_model


class TestModelTraining:
    """Tests for model training functions."""

    @pytest.fixture
    def sample_data(self):
        """Create sample classification data."""
        X, y = make_classification(
            n_samples=100,
            n_features=10,
            n_informative=5,
            n_redundant=2,
            random_state=42,
        )
        return X, y

    def test_get_model_logistic_regression(self):
        """Test creating a logistic regression model."""
        model = get_model("logistic_regression")
        assert model is not None
        assert hasattr(model, "fit")
        assert hasattr(model, "predict")

    def test_get_model_with_params(self):
        """Test creating a model with custom parameters."""
        params = {"C": 0.5, "max_iter": 100}
        model = get_model("logistic_regression", params)
        assert model.C == 0.5
        assert model.max_iter == 100

    def test_get_model_unknown(self):
        """Test that unknown model raises error."""
        with pytest.raises(ValueError, match="Unknown model"):
            get_model("unknown_model")

    def test_train_model(self, sample_data):
        """Test model training."""
        X, y = sample_data
        model = get_model("logistic_regression")

        trained = train_model(model, X, y)

        assert hasattr(trained, "coef_")
        assert trained.score(X, y) > 0.5

    def test_train_model_with_validation(self, sample_data):
        """Test model training with validation set."""
        X, y = sample_data

        # Split into train and val
        X_train, X_val = X[:80], X[80:]
        y_train, y_val = y[:80], y[80:]

        model = get_model("logistic_regression")
        trained = train_model(model, X_train, y_train, X_val, y_val)

        assert hasattr(trained, "coef_")

    def test_save_and_load_model(self, sample_data):
        """Test model serialization."""
        X, y = sample_data
        model = get_model("logistic_regression")
        model = train_model(model, X, y)

        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "test_model"

            # Save
            metadata = {"test": "value"}
            save_model(model, metadata, model_path)

            # Check files exist
            assert (model_path / "model.joblib").exists()
            assert (model_path / "metadata.json").exists()

            # Load
            loaded = load_model(model_path)

            # Verify same predictions
            original_pred = model.predict(X)
            loaded_pred = loaded.predict(X)
            assert np.array_equal(original_pred, loaded_pred)


class TestModelPrediction:
    """Tests for model prediction functions."""

    @pytest.fixture
    def trained_model(self):
        """Create a trained model for testing."""
        X, y = make_classification(n_samples=100, n_features=10, random_state=42)
        model = get_model("logistic_regression")
        model.fit(X, y)
        return model, X

    def test_predict_returns_binary(self, trained_model):
        """Test that predictions are binary."""
        model, X = trained_model
        predictions = model.predict(X)

        assert set(np.unique(predictions)).issubset({0, 1})

    def test_predict_proba_returns_valid_probabilities(self, trained_model):
        """Test that probabilities are valid."""
        model, X = trained_model
        probas = model.predict_proba(X)

        # Should have 2 columns (binary classification)
        assert probas.shape[1] == 2

        # Probabilities should sum to 1
        assert np.allclose(probas.sum(axis=1), 1.0)

        # Probabilities should be between 0 and 1
        assert (probas >= 0).all() and (probas <= 1).all()
