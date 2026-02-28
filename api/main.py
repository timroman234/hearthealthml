"""FastAPI application for HeartHealthML predictions."""

import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from api.models import (
    ErrorResponse,
    HealthResponse,
    PatientFeatures,
    PredictionResponse,
)
from src.features.build_features import engineer_features
from src.models.predict import load_production_model, predict_single
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Global model state
_model_state = {
    "model": None,
    "preprocessor": None,
    "version": None,
    "threshold": 0.5,
}


def load_model():
    """Load the production model into memory."""
    try:
        model, preprocessor = load_production_model()
        _model_state["model"] = model
        _model_state["preprocessor"] = preprocessor
        _model_state["version"] = "latest"
        logger.info("Model loaded successfully")
        return True
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return False


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler for startup/shutdown."""
    # Startup: Load model
    logger.info("Starting HeartHealthML API...")
    load_model()
    yield
    # Shutdown: Cleanup
    logger.info("Shutting down HeartHealthML API...")


app = FastAPI(
    title="HeartHealthML API",
    description="Heart disease prediction API using machine learning",
    version="0.1.0",
    lifespan=lifespan,
    responses={
        500: {"model": ErrorResponse, "description": "Internal server error"},
    },
)


@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["System"],
    summary="Health check endpoint",
)
async def health_check():
    """Check API health and model status."""
    model_loaded = _model_state["model"] is not None
    return HealthResponse(
        status="healthy" if model_loaded else "unhealthy",
        model_loaded=model_loaded,
        model_version=_model_state["version"],
    )


@app.post(
    "/predict",
    response_model=PredictionResponse,
    tags=["Predictions"],
    summary="Make a heart disease prediction",
    responses={
        503: {"model": ErrorResponse, "description": "Model not loaded"},
    },
)
async def predict(patient: PatientFeatures):
    """
    Make a heart disease prediction for a single patient.

    The model predicts the probability of heart disease based on
    clinical features. Returns a binary prediction, probability,
    and risk level (low/medium/high).
    """
    if _model_state["model"] is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please check /health endpoint.",
        )

    try:
        # Convert to dict and apply feature engineering
        import pandas as pd

        patient_dict = patient.model_dump()
        df = pd.DataFrame([patient_dict])

        # Apply feature engineering (same as training)
        df = engineer_features(df)

        # Convert back to dict for prediction
        engineered_patient = df.iloc[0].to_dict()

        # Make prediction
        result = predict_single(
            model=_model_state["model"],
            preprocessor=_model_state["preprocessor"],
            patient_data=engineered_patient,
            threshold=_model_state["threshold"],
        )

        return PredictionResponse(**result)

    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}",
        )


@app.post(
    "/reload",
    tags=["System"],
    summary="Reload the model",
    responses={
        503: {"model": ErrorResponse, "description": "Model reload failed"},
    },
)
async def reload_model():
    """Reload the production model from disk."""
    success = load_model()
    if not success:
        raise HTTPException(
            status_code=503,
            detail="Failed to reload model. Check logs for details.",
        )
    return {
        "message": "Model reloaded successfully",
        "version": _model_state["version"],
    }
