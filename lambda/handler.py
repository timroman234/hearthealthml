"""AWS Lambda handler for HeartHealthML serverless inference.

This Lambda function loads the model from S3 and makes predictions.
The model is cached between invocations for better performance.

Environment Variables:
    MODEL_BUCKET: S3 bucket containing model artifacts
    MODEL_KEY: S3 key for model file
    PREPROCESSOR_KEY: S3 key for preprocessor file

Usage:
    Deploy to AWS Lambda with API Gateway trigger for REST API access.
"""

import json
import os
from io import BytesIO

import boto3
import joblib
import pandas as pd

# Initialize S3 client
s3 = boto3.client("s3")

# Configuration from environment
BUCKET = os.environ.get("MODEL_BUCKET", "hearthealthml-artifacts")
MODEL_KEY = os.environ.get(
    "MODEL_KEY", "models/logistic_regression_v1.0.3/model.joblib"
)
PREPROCESSOR_KEY = os.environ.get(
    "PREPROCESSOR_KEY", "models/logistic_regression_v1.0.3/preprocessor.joblib"
)
THRESHOLD = float(os.environ.get("THRESHOLD", "0.5"))

# Global variables for model caching (persists between Lambda invocations)
_model = None
_preprocessor = None


def load_model():
    """Load model and preprocessor from S3.

    Models are cached globally to persist between Lambda invocations,
    reducing cold start latency.

    Returns:
        Tuple of (model, preprocessor).
    """
    global _model, _preprocessor

    if _model is None:
        print(f"Loading model from s3://{BUCKET}/{MODEL_KEY}")
        response = s3.get_object(Bucket=BUCKET, Key=MODEL_KEY)
        _model = joblib.load(BytesIO(response["Body"].read()))

        print(f"Loading preprocessor from s3://{BUCKET}/{PREPROCESSOR_KEY}")
        response = s3.get_object(Bucket=BUCKET, Key=PREPROCESSOR_KEY)
        _preprocessor = joblib.load(BytesIO(response["Body"].read()))

        print("Model loaded successfully")

    return _model, _preprocessor


def calculate_risk_level(probability: float) -> str:
    """Calculate risk level from probability.

    Args:
        probability: Prediction probability.

    Returns:
        Risk level string: 'low', 'medium', or 'high'.
    """
    if probability >= 0.7:
        return "high"
    elif probability >= 0.3:
        return "medium"
    return "low"


def validate_input(data: dict) -> list[str]:
    """Validate input data.

    Args:
        data: Input patient data.

    Returns:
        List of validation errors (empty if valid).
    """
    required_fields = [
        "age",
        "sex",
        "cp",
        "trestbps",
        "chol",
        "fbs",
        "restecg",
        "thalach",
        "exang",
        "oldpeak",
        "slope",
        "ca",
        "thal",
    ]

    errors = []
    for field in required_fields:
        if field not in data:
            errors.append(f"Missing required field: {field}")

    return errors


def handler(event, context):
    """AWS Lambda handler for predictions.

    Args:
        event: Lambda event (API Gateway format or direct invocation).
        context: Lambda context.

    Returns:
        API Gateway response with prediction results.
    """
    try:
        # Parse request body
        if isinstance(event.get("body"), str):
            body = json.loads(event["body"])
        elif event.get("body"):
            body = event["body"]
        else:
            # Direct invocation (not through API Gateway)
            body = event

        # Validate input
        errors = validate_input(body)
        if errors:
            return {
                "statusCode": 400,
                "headers": {
                    "Content-Type": "application/json",
                    "Access-Control-Allow-Origin": "*",
                },
                "body": json.dumps({"error": "Validation failed", "details": errors}),
            }

        # Load model (cached between invocations)
        model, preprocessor = load_model()

        # Create DataFrame with input data
        df = pd.DataFrame([body])

        # Apply feature engineering (simplified for Lambda)
        # Note: Full feature engineering would require importing src.features
        # For Lambda, we assume preprocessor handles raw features
        df["heart_rate_reserve"] = (220 - df["age"]) - df["thalach"]
        df["cardiac_risk_score"] = (
            (df["age"] > 55).astype(int)
            + (df["sex"] == 1).astype(int)
            + (df["cp"] == 0).astype(int) * 2
            + (df["trestbps"] >= 140).astype(int)
            + (df["chol"] >= 240).astype(int)
            + (df["fbs"] == 1).astype(int)
            + (df["exang"] == 1).astype(int) * 2
            + (df["oldpeak"] > 2).astype(int) * 2
            + (df["ca"] > 0).astype(int) * 2
        )

        # Preprocess input
        X = preprocessor.transform(df)

        # Make prediction
        probability = float(model.predict_proba(X)[0, 1])
        prediction = int(probability >= THRESHOLD)
        risk_level = calculate_risk_level(probability)
        confidence = probability if prediction == 1 else (1 - probability)

        response = {
            "prediction": prediction,
            "prediction_label": "heart_disease" if prediction == 1 else "healthy",
            "probability": round(probability, 4),
            "risk_level": risk_level,
            "confidence": round(confidence, 4),
            "threshold_used": THRESHOLD,
        }

        return {
            "statusCode": 200,
            "headers": {
                "Content-Type": "application/json",
                "Access-Control-Allow-Origin": "*",
            },
            "body": json.dumps(response),
        }

    except json.JSONDecodeError as e:
        return {
            "statusCode": 400,
            "headers": {
                "Content-Type": "application/json",
                "Access-Control-Allow-Origin": "*",
            },
            "body": json.dumps({"error": f"Invalid JSON: {str(e)}"}),
        }
    except Exception as e:
        print(f"Error: {str(e)}")
        return {
            "statusCode": 500,
            "headers": {
                "Content-Type": "application/json",
                "Access-Control-Allow-Origin": "*",
            },
            "body": json.dumps({"error": str(e)}),
        }


def health_check(event, context):
    """Health check handler for Lambda.

    Args:
        event: Lambda event.
        context: Lambda context.

    Returns:
        Health status response.
    """
    try:
        model, _ = load_model()
        model_loaded = model is not None
    except Exception:
        model_loaded = False

    return {
        "statusCode": 200,
        "headers": {
            "Content-Type": "application/json",
            "Access-Control-Allow-Origin": "*",
        },
        "body": json.dumps(
            {
                "status": "healthy" if model_loaded else "unhealthy",
                "model_loaded": model_loaded,
                "model_key": MODEL_KEY,
            }
        ),
    }
