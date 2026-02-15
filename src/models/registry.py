"""Model registry and versioning."""

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import joblib
from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer

from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class ModelVersion:
    """Represents a versioned model."""

    name: str
    version: str
    path: Path
    metrics: dict
    metadata: dict
    created_at: datetime


class ModelRegistry:
    """Model registry for versioning and tracking model artifacts."""

    def __init__(self, registry_path: Path):
        """Initialize the model registry.

        Args:
            registry_path: Path to the registry directory.
        """
        self.registry_path = Path(registry_path)
        self.registry_path.mkdir(parents=True, exist_ok=True)
        self.registry_file = self.registry_path / "registry.json"
        self.registry = self._load_registry()

    def _load_registry(self) -> dict:
        """Load registry from disk."""
        if self.registry_file.exists():
            with open(self.registry_file) as f:
                return json.load(f)
        return {"models": {}}

    def _save_registry(self) -> None:
        """Save registry to disk."""
        with open(self.registry_file, "w") as f:
            json.dump(self.registry, f, indent=2, default=str)

    def _generate_version(self, model_name: str) -> str:
        """Generate next version number for a model."""
        if model_name not in self.registry["models"]:
            return "1.0.0"

        versions = self.registry["models"][model_name]
        if not versions:
            return "1.0.0"

        # Get latest version and increment
        latest = sorted(versions.keys())[-1]
        major, minor, patch = map(int, latest.split("."))
        return f"{major}.{minor}.{patch + 1}"

    def _get_latest_version(self, model_name: str) -> str:
        """Get latest version for a model."""
        if model_name not in self.registry["models"]:
            raise ValueError(f"Model not found: {model_name}")

        versions = self.registry["models"][model_name]
        if not versions:
            raise ValueError(f"No versions found for model: {model_name}")

        return sorted(versions.keys())[-1]

    def register_model(
        self,
        model: BaseEstimator,
        preprocessor: ColumnTransformer,
        metrics: dict,
        metadata: dict,
    ) -> str:
        """Register a new model version.

        Args:
            model: Trained model.
            preprocessor: Fitted preprocessor.
            metrics: Evaluation metrics.
            metadata: Training metadata.

        Returns:
            Version string for the registered model.
        """
        model_name = metadata.get("model_name", "unknown")
        version = self._generate_version(model_name)

        model_dir = self.registry_path / f"{model_name}_v{version}"
        model_dir.mkdir(parents=True, exist_ok=True)

        # Save model
        joblib.dump(model, model_dir / "model.joblib")

        # Save preprocessor
        joblib.dump(preprocessor, model_dir / "preprocessor.joblib")

        # Save metrics
        with open(model_dir / "metrics.json", "w") as f:
            json.dump(metrics, f, indent=2, default=str)

        # Save metadata
        metadata["version"] = version
        metadata["created_at"] = datetime.now().isoformat()
        with open(model_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2, default=str)

        # Update registry
        if model_name not in self.registry["models"]:
            self.registry["models"][model_name] = {}

        self.registry["models"][model_name][version] = {
            "path": str(model_dir),
            "created_at": metadata["created_at"],
            "metrics": metrics,
        }
        self._save_registry()

        logger.info(f"Registered {model_name} v{version} at {model_dir}")
        return version

    def load_model(
        self, model_name: str, version: str = "latest"
    ) -> tuple[BaseEstimator, ColumnTransformer]:
        """Load model and preprocessor by name and version.

        Args:
            model_name: Name of the model.
            version: Version string or 'latest'.

        Returns:
            Tuple of (model, preprocessor).
        """
        if version == "latest":
            version = self._get_latest_version(model_name)

        model_dir = self.registry_path / f"{model_name}_v{version}"

        if not model_dir.exists():
            raise ValueError(f"Model not found: {model_name} v{version}")

        model = joblib.load(model_dir / "model.joblib")
        preprocessor = joblib.load(model_dir / "preprocessor.joblib")

        logger.info(f"Loaded {model_name} v{version}")
        return model, preprocessor

    def load_metrics(self, model_name: str, version: str = "latest") -> dict:
        """Load metrics for a model version.

        Args:
            model_name: Name of the model.
            version: Version string or 'latest'.

        Returns:
            Metrics dictionary.
        """
        if version == "latest":
            version = self._get_latest_version(model_name)

        model_dir = self.registry_path / f"{model_name}_v{version}"
        with open(model_dir / "metrics.json") as f:
            return json.load(f)

    def load_metadata(self, model_name: str, version: str = "latest") -> dict:
        """Load metadata for a model version.

        Args:
            model_name: Name of the model.
            version: Version string or 'latest'.

        Returns:
            Metadata dictionary.
        """
        if version == "latest":
            version = self._get_latest_version(model_name)

        model_dir = self.registry_path / f"{model_name}_v{version}"
        with open(model_dir / "metadata.json") as f:
            return json.load(f)

    def list_models(self) -> list[str]:
        """List all registered model names."""
        return list(self.registry["models"].keys())

    def list_versions(self, model_name: str) -> list[str]:
        """List all versions for a model."""
        if model_name not in self.registry["models"]:
            return []
        return sorted(self.registry["models"][model_name].keys())

    def get_model_info(self, model_name: str, version: str = "latest") -> dict:
        """Get info for a specific model version.

        Args:
            model_name: Name of the model.
            version: Version string or 'latest'.

        Returns:
            Dictionary with model info.
        """
        if version == "latest":
            version = self._get_latest_version(model_name)

        return self.registry["models"][model_name][version]
