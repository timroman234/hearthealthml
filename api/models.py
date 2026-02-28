"""Pydantic schemas for API request/response validation."""

from typing import Literal

from pydantic import BaseModel, Field


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
                    "age": 63,
                    "sex": 1,
                    "cp": 3,
                    "trestbps": 145,
                    "chol": 233,
                    "fbs": 1,
                    "restecg": 0,
                    "thalach": 150,
                    "exang": 0,
                    "oldpeak": 2.3,
                    "slope": 0,
                    "ca": 0,
                    "thal": 1,
                }
            ]
        }
    }


class PredictionResponse(BaseModel):
    """Response schema for prediction endpoint."""

    prediction: Literal[0, 1] = Field(
        ..., description="Binary prediction (0=healthy, 1=heart disease)"
    )
    prediction_label: Literal["healthy", "heart_disease"] = Field(
        ..., description="Human-readable prediction label"
    )
    probability: float = Field(
        ..., ge=0.0, le=1.0, description="Probability of heart disease"
    )
    risk_level: Literal["low", "medium", "high"] = Field(
        ..., description="Risk level category"
    )
    threshold_used: float = Field(..., description="Classification threshold used")


class HealthResponse(BaseModel):
    """Response schema for health check endpoint."""

    status: Literal["healthy", "unhealthy"] = Field(..., description="Service status")
    model_loaded: bool = Field(..., description="Whether model is loaded")
    model_version: str | None = Field(None, description="Loaded model version")


class ErrorResponse(BaseModel):
    """Response schema for error cases."""

    detail: str = Field(..., description="Error message")
