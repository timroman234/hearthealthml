"""Dependency injection for FastAPI application."""

import os
import time
from functools import lru_cache
from pathlib import Path
from typing import Any

from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer

from src.models.registry import ModelRegistry
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Startup time for uptime calculation
START_TIME = time.time()


class ModelLoader:
    """Singleton model loader with caching."""

    _instance = None
    _model: BaseEstimator | None = None
    _preprocessor: ColumnTransformer | None = None
    _metadata: dict[str, Any] | None = None
    _version: str | None = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def load(
        self, model_name: str | None = None, version: str | None = None
    ) -> tuple[BaseEstimator, ColumnTransformer, dict[str, Any]]:
        """Load model, preprocessor, and metadata.

        Args:
            model_name: Name of the model to load.
            version: Version string or 'latest'.

        Returns:
            Tuple of (model, preprocessor, metadata).
        """
        if self._model is not None:
            return self._model, self._preprocessor, self._metadata

        model_name = model_name or os.getenv("MODEL_NAME", "logistic_regression")
        version = version or os.getenv("MODEL_VERSION", "latest")

        # Load from local registry
        registry_path = Path(os.getenv("MODEL_REGISTRY_PATH", "models/"))
        registry = ModelRegistry(registry_path)

        self._model, self._preprocessor = registry.load_model(model_name, version)
        self._metadata = registry.get_model_info(model_name, version) or {}

        # Get actual version if 'latest' was specified
        if version == "latest":
            self._version = registry._get_latest_version(model_name)
        else:
            self._version = version

        logger.info(f"Loaded model: {model_name} v{self._version}")
        return self._model, self._preprocessor, self._metadata

    def reload(
        self, model_name: str | None = None, version: str | None = None
    ) -> tuple[BaseEstimator, ColumnTransformer, dict[str, Any]]:
        """Force reload the model.

        Args:
            model_name: Name of the model to load.
            version: Version string or 'latest'.

        Returns:
            Tuple of (model, preprocessor, metadata).
        """
        self._model = None
        self._preprocessor = None
        self._metadata = None
        self._version = None
        return self.load(model_name, version)

    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._model is not None

    @property
    def version(self) -> str:
        """Get loaded model version."""
        return self._version or "unknown"

    @property
    def model(self) -> BaseEstimator | None:
        """Get loaded model."""
        return self._model

    @property
    def preprocessor(self) -> ColumnTransformer | None:
        """Get loaded preprocessor."""
        return self._preprocessor

    @property
    def metadata(self) -> dict[str, Any]:
        """Get model metadata."""
        return self._metadata or {}


@lru_cache()
def get_model_loader() -> ModelLoader:
    """Get singleton model loader instance.

    Returns:
        ModelLoader instance with model loaded.
    """
    loader = ModelLoader()
    try:
        loader.load()
    except Exception as e:
        logger.warning(f"Failed to load model on startup: {e}")
    return loader


def get_uptime() -> float:
    """Get server uptime in seconds.

    Returns:
        Uptime in seconds since server start.
    """
    return time.time() - START_TIME
