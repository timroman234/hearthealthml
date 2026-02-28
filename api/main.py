"""FastAPI application for HeartHealthML predictions."""

import os
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from starlette.responses import Response

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from api.dependencies import ModelLoader, get_model_loader, get_uptime
from api.models import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    ErrorResponse,
    HealthResponse,
    ModelInfoResponse,
    PatientFeatures,
    PredictionResponse,
    RiskLevel,
)
from src.features.build_features import engineer_features
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Prometheus metrics
PREDICTIONS_TOTAL = Counter(
    "hearthealthml_predictions_total",
    "Total number of predictions made",
    ["risk_level"],
)
PREDICTION_LATENCY = Histogram(
    "hearthealthml_prediction_latency_seconds",
    "Time spent processing prediction requests",
    buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0],
)
REQUESTS_TOTAL = Counter(
    "hearthealthml_requests_total",
    "Total HTTP requests",
    ["method", "endpoint", "status"],
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler for startup/shutdown."""
    # Startup: Load model
    logger.info("Starting HeartHealthML API...")
    loader = get_model_loader()
    if loader.is_loaded:
        logger.info(f"Model loaded: version {loader.version}")
    else:
        logger.warning("Model not loaded - check model files")
    yield
    # Shutdown: Cleanup
    logger.info("Shutting down HeartHealthML API...")


# Create FastAPI app
app = FastAPI(
    title="HeartHealthML API",
    description="Heart disease prediction API using machine learning",
    version="1.0.0",
    lifespan=lifespan,
    responses={
        500: {"model": ErrorResponse, "description": "Internal server error"},
    },
)

# CORS middleware for cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def track_requests(request: Request, call_next):
    """Track request metrics for Prometheus."""
    start_time = time.time()
    response = await call_next(request)

    # Track request metrics
    REQUESTS_TOTAL.labels(
        method=request.method,
        endpoint=request.url.path,
        status=response.status_code,
    ).inc()

    # Log slow requests
    duration = time.time() - start_time
    if duration > 1.0:
        logger.warning(
            f"Slow request: {request.method} {request.url.path} ({duration:.2f}s)"
        )

    return response


def calculate_risk_level(probability: float) -> RiskLevel:
    """Calculate risk level from probability.

    Args:
        probability: Prediction probability.

    Returns:
        RiskLevel enum value.
    """
    if probability >= 0.7:
        return RiskLevel.HIGH
    elif probability >= 0.3:
        return RiskLevel.MEDIUM
    return RiskLevel.LOW


def preprocess_patient(patient: PatientFeatures, preprocessor) -> np.ndarray:
    """Preprocess patient data for prediction.

    Args:
        patient: Patient features.
        preprocessor: Fitted preprocessor.

    Returns:
        Preprocessed feature array.
    """
    # Convert to DataFrame
    df = pd.DataFrame([patient.model_dump()])

    # Apply feature engineering
    df = engineer_features(df)

    # Apply preprocessing
    return preprocessor.transform(df)


@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check(loader: ModelLoader = Depends(get_model_loader)):
    """Health check endpoint.

    Returns the current health status of the API including model status and uptime.
    """
    return HealthResponse(
        status="healthy" if loader.is_loaded else "unhealthy",
        model_loaded=loader.is_loaded,
        model_version=loader.version if loader.is_loaded else None,
        uptime_seconds=get_uptime(),
    )


@app.get("/metrics", tags=["Monitoring"])
async def metrics():
    """Prometheus metrics endpoint.

    Returns metrics in Prometheus text format for scraping.
    """
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/model-info", response_model=ModelInfoResponse, tags=["Model"])
async def model_info(loader: ModelLoader = Depends(get_model_loader)):
    """Get model information.

    Returns details about the currently loaded model including version,
    features, performance metrics, and classification threshold.
    """
    if not loader.is_loaded:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please check /health endpoint.",
        )

    metadata = loader.metadata

    # Get feature names from preprocessor if available
    features = metadata.get("features_used", [])
    if not features and loader.preprocessor is not None:
        try:
            features = list(loader.preprocessor.feature_names_in_)
        except AttributeError:
            features = []

    return ModelInfoResponse(
        model_name=metadata.get("model_name", "logistic_regression"),
        version=loader.version,
        features=features,
        metrics=metadata.get("metrics", {}),
        threshold=metadata.get("optimal_threshold", 0.5),
    )


@app.post(
    "/predict",
    response_model=PredictionResponse,
    tags=["Predictions"],
    responses={
        503: {"model": ErrorResponse, "description": "Model not loaded"},
    },
)
async def predict(
    patient: PatientFeatures,
    loader: ModelLoader = Depends(get_model_loader),
):
    """Make a prediction for a single patient.

    The model predicts the probability of heart disease based on
    clinical features. Returns a binary prediction, probability,
    risk level, and model confidence.
    """
    if not loader.is_loaded:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please check /health endpoint.",
        )

    with PREDICTION_LATENCY.time():
        try:
            model = loader.model
            preprocessor = loader.preprocessor
            threshold = loader.metadata.get("optimal_threshold", 0.5)

            # Preprocess patient data
            X = preprocess_patient(patient, preprocessor)

            # Make prediction
            probability = float(model.predict_proba(X)[0, 1])
            prediction = int(probability >= threshold)
            risk_level = calculate_risk_level(probability)
            confidence = probability if prediction == 1 else (1 - probability)

            # Track metrics
            PREDICTIONS_TOTAL.labels(risk_level=risk_level.value).inc()

            return PredictionResponse(
                prediction=prediction,
                prediction_label="heart_disease" if prediction == 1 else "healthy",
                probability=probability,
                risk_level=risk_level,
                confidence=confidence,
                threshold_used=threshold,
            )

        except Exception as e:
            logger.error(f"Prediction error: {e}")
            raise HTTPException(status_code=500, detail=str(e))


@app.post(
    "/predict/batch",
    response_model=BatchPredictionResponse,
    tags=["Predictions"],
    responses={
        503: {"model": ErrorResponse, "description": "Model not loaded"},
    },
)
async def predict_batch(
    request: BatchPredictionRequest,
    loader: ModelLoader = Depends(get_model_loader),
):
    """Make predictions for multiple patients.

    Accepts a batch of patient records and returns predictions for all.
    Maximum batch size is 100 patients.
    """
    if not loader.is_loaded:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please check /health endpoint.",
        )

    predictions = []
    high_risk_count = 0

    for patient in request.patients:
        try:
            result = await predict(patient, loader)
            predictions.append(result)
            if result.risk_level == RiskLevel.HIGH:
                high_risk_count += 1
        except HTTPException:
            # Add error placeholder for failed predictions
            predictions.append(
                PredictionResponse(
                    prediction=-1,
                    prediction_label="error",
                    probability=0.0,
                    risk_level=RiskLevel.LOW,
                    confidence=0.0,
                    threshold_used=0.5,
                )
            )

    return BatchPredictionResponse(
        predictions=predictions,
        total_count=len(predictions),
        high_risk_count=high_risk_count,
    )


@app.post("/reload", tags=["System"])
async def reload_model(loader: ModelLoader = Depends(get_model_loader)):
    """Reload the production model from disk.

    Forces a reload of the model even if already loaded.
    Useful after deploying a new model version.
    """
    try:
        loader.reload()
        logger.info(f"Model reloaded: version {loader.version}")
        return {
            "message": "Model reloaded successfully",
            "version": loader.version,
        }
    except Exception as e:
        logger.error(f"Failed to reload model: {e}")
        raise HTTPException(
            status_code=503,
            detail=f"Failed to reload model: {str(e)}",
        )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "api.main:app",
        host=os.getenv("API_HOST", "0.0.0.0"),
        port=int(os.getenv("API_PORT", 8000)),
        reload=True,
    )
