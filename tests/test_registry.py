"""Tests for model registry."""

import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.registry import ModelRegistry


class TestModelRegistry:
    """Tests for ModelRegistry class."""

    @pytest.fixture
    def temp_registry(self):
        """Create a temporary registry directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield ModelRegistry(Path(tmpdir))

    @pytest.fixture
    def sample_model(self):
        """Create a sample trained model."""
        np.random.seed(42)
        X = np.random.randn(50, 3)
        y = (X[:, 0] > 0).astype(int)
        model = LogisticRegression(random_state=42)
        model.fit(X, y)
        return model

    @pytest.fixture
    def sample_preprocessor(self):
        """Create a sample preprocessor."""
        preprocessor = ColumnTransformer(
            transformers=[("scaler", StandardScaler(), [0, 1, 2])],
            remainder="drop",
        )
        X = np.random.randn(50, 3)
        preprocessor.fit(X)
        return preprocessor

    def test_init_creates_directory(self):
        """Test that init creates the registry directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            registry_path = Path(tmpdir) / "new_registry"
            ModelRegistry(registry_path)

            assert registry_path.exists()

    def test_register_model(self, temp_registry, sample_model, sample_preprocessor):
        """Test registering a model."""
        metrics = {"accuracy": 0.85, "f1": 0.82}
        metadata = {"model_name": "test_model", "features": ["a", "b", "c"]}

        version = temp_registry.register_model(
            sample_model, sample_preprocessor, metrics, metadata
        )

        assert version == "1.0.0"

    def test_register_model_increments_version(
        self, temp_registry, sample_model, sample_preprocessor
    ):
        """Test that version increments on subsequent registrations."""
        metrics = {"accuracy": 0.85}
        metadata = {"model_name": "test_model"}

        v1 = temp_registry.register_model(
            sample_model, sample_preprocessor, metrics, metadata
        )
        v2 = temp_registry.register_model(
            sample_model, sample_preprocessor, metrics, metadata
        )
        v3 = temp_registry.register_model(
            sample_model, sample_preprocessor, metrics, metadata
        )

        assert v1 == "1.0.0"
        assert v2 == "1.0.1"
        assert v3 == "1.0.2"

    def test_load_model(self, temp_registry, sample_model, sample_preprocessor):
        """Test loading a registered model."""
        metrics = {"accuracy": 0.85}
        metadata = {"model_name": "test_model"}

        version = temp_registry.register_model(
            sample_model, sample_preprocessor, metrics, metadata
        )

        loaded_model, loaded_preprocessor = temp_registry.load_model(
            "test_model", version
        )

        assert loaded_model is not None
        assert loaded_preprocessor is not None

    def test_load_model_latest(self, temp_registry, sample_model, sample_preprocessor):
        """Test loading latest model version."""
        metrics = {"accuracy": 0.85}
        metadata = {"model_name": "test_model"}

        # Register multiple versions
        temp_registry.register_model(
            sample_model, sample_preprocessor, metrics, metadata
        )
        temp_registry.register_model(
            sample_model, sample_preprocessor, metrics, metadata
        )

        loaded_model, loaded_preprocessor = temp_registry.load_model(
            "test_model", "latest"
        )

        assert loaded_model is not None

    def test_load_model_not_found(self, temp_registry):
        """Test error when model doesn't exist."""
        with pytest.raises(ValueError, match="Model not found"):
            temp_registry.load_model("nonexistent", "1.0.0")

    def test_load_model_version_not_found(
        self, temp_registry, sample_model, sample_preprocessor
    ):
        """Test error when version doesn't exist."""
        metrics = {"accuracy": 0.85}
        metadata = {"model_name": "test_model"}

        temp_registry.register_model(
            sample_model, sample_preprocessor, metrics, metadata
        )

        with pytest.raises(ValueError, match="Model not found"):
            temp_registry.load_model("test_model", "9.9.9")

    def test_load_metrics(self, temp_registry, sample_model, sample_preprocessor):
        """Test loading model metrics."""
        metrics = {"accuracy": 0.85, "f1": 0.82}
        metadata = {"model_name": "test_model"}

        version = temp_registry.register_model(
            sample_model, sample_preprocessor, metrics, metadata
        )

        loaded_metrics = temp_registry.load_metrics("test_model", version)

        assert loaded_metrics["accuracy"] == 0.85
        assert loaded_metrics["f1"] == 0.82

    def test_load_metrics_latest(
        self, temp_registry, sample_model, sample_preprocessor
    ):
        """Test loading metrics for latest version."""
        metadata = {"model_name": "test_model"}

        temp_registry.register_model(
            sample_model, sample_preprocessor, {"accuracy": 0.80}, metadata
        )
        temp_registry.register_model(
            sample_model, sample_preprocessor, {"accuracy": 0.85}, metadata
        )

        loaded_metrics = temp_registry.load_metrics("test_model", "latest")

        assert loaded_metrics["accuracy"] == 0.85

    def test_load_metadata(self, temp_registry, sample_model, sample_preprocessor):
        """Test loading model metadata."""
        metrics = {"accuracy": 0.85}
        metadata = {"model_name": "test_model", "custom_field": "value"}

        version = temp_registry.register_model(
            sample_model, sample_preprocessor, metrics, metadata
        )

        loaded_metadata = temp_registry.load_metadata("test_model", version)

        assert loaded_metadata["model_name"] == "test_model"
        assert loaded_metadata["custom_field"] == "value"
        assert "version" in loaded_metadata
        assert "created_at" in loaded_metadata

    def test_load_metadata_latest(
        self, temp_registry, sample_model, sample_preprocessor
    ):
        """Test loading metadata for latest version."""
        metrics = {"accuracy": 0.85}
        metadata = {"model_name": "test_model"}

        temp_registry.register_model(
            sample_model, sample_preprocessor, metrics, metadata
        )

        loaded_metadata = temp_registry.load_metadata("test_model", "latest")

        assert loaded_metadata is not None

    def test_list_models_empty(self, temp_registry):
        """Test listing models when registry is empty."""
        models = temp_registry.list_models()

        assert models == []

    def test_list_models(self, temp_registry, sample_model, sample_preprocessor):
        """Test listing registered models."""
        metrics = {"accuracy": 0.85}

        temp_registry.register_model(
            sample_model, sample_preprocessor, metrics, {"model_name": "model_a"}
        )
        temp_registry.register_model(
            sample_model, sample_preprocessor, metrics, {"model_name": "model_b"}
        )

        models = temp_registry.list_models()

        assert "model_a" in models
        assert "model_b" in models

    def test_list_versions_empty(self, temp_registry):
        """Test listing versions for nonexistent model."""
        versions = temp_registry.list_versions("nonexistent")

        assert versions == []

    def test_list_versions(self, temp_registry, sample_model, sample_preprocessor):
        """Test listing model versions."""
        metrics = {"accuracy": 0.85}
        metadata = {"model_name": "test_model"}

        temp_registry.register_model(
            sample_model, sample_preprocessor, metrics, metadata
        )
        temp_registry.register_model(
            sample_model, sample_preprocessor, metrics, metadata
        )
        temp_registry.register_model(
            sample_model, sample_preprocessor, metrics, metadata
        )

        versions = temp_registry.list_versions("test_model")

        assert versions == ["1.0.0", "1.0.1", "1.0.2"]

    def test_get_model_info(self, temp_registry, sample_model, sample_preprocessor):
        """Test getting model info."""
        metrics = {"accuracy": 0.85}
        metadata = {"model_name": "test_model"}

        version = temp_registry.register_model(
            sample_model, sample_preprocessor, metrics, metadata
        )

        info = temp_registry.get_model_info("test_model", version)

        assert "path" in info
        assert "created_at" in info
        assert "metrics" in info
        assert info["metrics"]["accuracy"] == 0.85

    def test_get_model_info_latest(
        self, temp_registry, sample_model, sample_preprocessor
    ):
        """Test getting info for latest version."""
        metrics = {"accuracy": 0.85}
        metadata = {"model_name": "test_model"}

        temp_registry.register_model(
            sample_model, sample_preprocessor, metrics, metadata
        )

        info = temp_registry.get_model_info("test_model", "latest")

        assert info is not None

    def test_get_latest_version_no_versions(self, temp_registry):
        """Test error when getting latest version for model with no versions."""
        with pytest.raises(ValueError, match="Model not found"):
            temp_registry._get_latest_version("nonexistent")

    def test_registry_persists(self, sample_model, sample_preprocessor):
        """Test that registry data persists across instances."""
        with tempfile.TemporaryDirectory() as tmpdir:
            registry_path = Path(tmpdir)

            # Create and register with first instance
            registry1 = ModelRegistry(registry_path)
            metrics = {"accuracy": 0.85}
            metadata = {"model_name": "test_model"}
            registry1.register_model(
                sample_model, sample_preprocessor, metrics, metadata
            )

            # Create new instance and verify data persists
            registry2 = ModelRegistry(registry_path)
            models = registry2.list_models()

            assert "test_model" in models

    def test_model_predictions_consistent(
        self, temp_registry, sample_model, sample_preprocessor
    ):
        """Test that loaded model produces same predictions."""
        metrics = {"accuracy": 0.85}
        metadata = {"model_name": "test_model"}

        # Get predictions from original model
        X_test = np.random.randn(10, 3)
        original_pred = sample_model.predict(X_test)

        # Register and load
        version = temp_registry.register_model(
            sample_model, sample_preprocessor, metrics, metadata
        )
        loaded_model, _ = temp_registry.load_model("test_model", version)

        # Compare predictions
        loaded_pred = loaded_model.predict(X_test)

        assert np.array_equal(original_pred, loaded_pred)
