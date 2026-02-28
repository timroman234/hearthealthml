"""Pydantic schemas for API request/response validation."""

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field


class RiskLevel(str, Enum):
    """Risk level categorization."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class PatientFeatures(BaseModel):
    """Input features for heart disease prediction.

    All features correspond to the UCI Heart Disease dataset.
    """

    # Continuous features
    age: int = Field(..., ge=1, le=120, description="Age in years")
    trestbps: int = Field(
        ..., ge=80, le=220, description="Resting blood pressure (mm Hg)"
    )
    chol: int = Field(..., ge=100, le=600, description="Serum cholesterol (mg/dl)")
    thalach: int = Field(..., ge=60, le=220, description="Maximum heart rate achieved")
    oldpeak: float = Field(
        ..., ge=0.0, le=10.0, description="ST depression induced by exercise"
    )

    # Binary features
    sex: Literal[0, 1] = Field(..., description="Sex (0=female, 1=male)")
    fbs: Literal[0, 1] = Field(
        ..., description="Fasting blood sugar > 120 mg/dl (0=false, 1=true)"
    )
    exang: Literal[0, 1] = Field(
        ..., description="Exercise induced angina (0=no, 1=yes)"
    )

    # Categorical features
    cp: Literal[0, 1, 2, 3] = Field(
        ...,
        description=(
            "Chest pain type: 0=typical angina, 1=atypical angina, "
            "2=non-anginal pain, 3=asymptomatic"
        ),
    )
    restecg: Literal[0, 1, 2] = Field(
        ...,
        description=(
            "Resting ECG: 0=normal, 1=ST-T wave abnormality, "
            "2=left ventricular hypertrophy"
        ),
    )
    slope: Literal[0, 1, 2] = Field(
        ...,
        description="Slope of peak exercise ST segment: 0=upsloping, 1=flat, 2=downsloping",
    )
    ca: Literal[0, 1, 2, 3, 4] = Field(
        ..., description="Number of major vessels colored by fluoroscopy (0-4)"
    )
    thal: Literal[0, 1, 2, 3] = Field(
        ...,
        description="Thalassemia: 0=unknown, 1=normal, 2=fixed defect, 3=reversible defect",
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "age": 55,
                    "sex": 1,
                    "cp": 0,
                    "trestbps": 140,
                    "chol": 250,
                    "fbs": 0,
                    "restecg": 1,
                    "thalach": 150,
                    "exang": 0,
                    "oldpeak": 1.5,
                    "slope": 1,
                    "ca": 0,
                    "thal": 2,
                }
            ]
        }
    }


class PredictionResponse(BaseModel):
    """Response schema for prediction endpoint."""

    prediction: int = Field(
        ..., ge=-1, le=1, description="Binary prediction (0=healthy, 1=heart disease)"
    )
    prediction_label: str = Field(..., description="Human-readable prediction label")
    probability: float = Field(
        ..., ge=0.0, le=1.0, description="Probability of heart disease"
    )
    risk_level: RiskLevel = Field(..., description="Risk level category")
    confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Model confidence in prediction"
    )
    threshold_used: float = Field(default=0.5, description="Classification threshold")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "prediction": 1,
                    "prediction_label": "heart_disease",
                    "probability": 0.73,
                    "risk_level": "high",
                    "confidence": 0.73,
                    "threshold_used": 0.5,
                }
            ]
        }
    }


class BatchPredictionRequest(BaseModel):
    """Input schema for batch predictions."""

    patients: list[PatientFeatures] = Field(
        ..., min_length=1, max_length=100, description="List of patient records"
    )


class BatchPredictionResponse(BaseModel):
    """Output schema for batch predictions."""

    predictions: list[PredictionResponse] = Field(..., description="Prediction results")
    total_count: int = Field(..., description="Total number of predictions")
    high_risk_count: int = Field(..., description="Number of high-risk predictions")


class HealthResponse(BaseModel):
    """Response schema for health check endpoint."""

    status: Literal["healthy", "unhealthy"] = Field(..., description="Service status")
    model_loaded: bool = Field(..., description="Whether model is loaded")
    model_version: str | None = Field(None, description="Loaded model version")
    uptime_seconds: float = Field(..., description="Server uptime in seconds")


class ModelInfoResponse(BaseModel):
    """Response schema for model information endpoint."""

    model_name: str = Field(..., description="Name of the loaded model")
    version: str = Field(..., description="Model version")
    features: list[str] = Field(..., description="List of input features")
    metrics: dict = Field(..., description="Model performance metrics")
    threshold: float = Field(..., description="Classification threshold")


class ErrorResponse(BaseModel):
    """Response schema for error cases."""

    detail: str = Field(..., description="Error message")
