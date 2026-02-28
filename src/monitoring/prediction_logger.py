"""Structured logging for predictions and model events."""

import json
import logging
import sys
from datetime import datetime
from typing import Any


class JSONFormatter(logging.Formatter):
    """JSON log formatter for structured logging.

    Outputs logs in JSON format for easy parsing by log aggregation
    systems like ELK, CloudWatch, or Datadog.
    """

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON.

        Args:
            record: Log record to format.

        Returns:
            JSON string representation of the log.
        """
        log_data = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        # Add extra fields if present
        if hasattr(record, "extra") and isinstance(record.extra, dict):
            log_data.update(record.extra)

        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_data)


def setup_json_logger(
    name: str,
    level: int = logging.INFO,
) -> logging.Logger:
    """Set up a logger with JSON formatting.

    Args:
        name: Logger name.
        level: Logging level.

    Returns:
        Configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Remove existing handlers
    logger.handlers = []

    # Console handler with JSON formatting
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)
    handler.setFormatter(JSONFormatter())

    logger.addHandler(handler)

    return logger


class PredictionLogger:
    """Logger for tracking predictions with structured output.

    Provides methods for logging prediction events, model events,
    and performance metrics in a consistent format.
    """

    def __init__(
        self,
        logger: logging.Logger | None = None,
        json_format: bool = True,
    ):
        """Initialize the prediction logger.

        Args:
            logger: Logger instance to use. Creates new if None.
            json_format: Whether to use JSON formatting.
        """
        if logger is None:
            if json_format:
                self.logger = setup_json_logger("hearthealthml.predictions")
            else:
                self.logger = logging.getLogger("hearthealthml.predictions")
        else:
            self.logger = logger

    def log_prediction(
        self,
        patient_id: str | None,
        prediction: int,
        probability: float,
        risk_level: str,
        latency_ms: float,
        **extra: Any,
    ) -> None:
        """Log a prediction event.

        Args:
            patient_id: Optional patient identifier.
            prediction: Binary prediction result.
            probability: Prediction probability.
            risk_level: Risk level category.
            latency_ms: Prediction latency in milliseconds.
            **extra: Additional fields to log.
        """
        self.logger.info(
            "Prediction made",
            extra={
                "extra": {
                    "event_type": "prediction",
                    "patient_id": patient_id,
                    "prediction": prediction,
                    "prediction_label": (
                        "heart_disease" if prediction == 1 else "healthy"
                    ),
                    "probability": round(probability, 4),
                    "risk_level": risk_level,
                    "latency_ms": round(latency_ms, 2),
                    **extra,
                }
            },
        )

    def log_batch_prediction(
        self,
        batch_size: int,
        high_risk_count: int,
        total_latency_ms: float,
        **extra: Any,
    ) -> None:
        """Log a batch prediction event.

        Args:
            batch_size: Number of predictions in batch.
            high_risk_count: Number of high-risk predictions.
            total_latency_ms: Total batch latency in milliseconds.
            **extra: Additional fields to log.
        """
        self.logger.info(
            "Batch prediction completed",
            extra={
                "extra": {
                    "event_type": "batch_prediction",
                    "batch_size": batch_size,
                    "high_risk_count": high_risk_count,
                    "high_risk_rate": (
                        round(high_risk_count / batch_size, 4) if batch_size > 0 else 0
                    ),
                    "total_latency_ms": round(total_latency_ms, 2),
                    "avg_latency_ms": (
                        round(total_latency_ms / batch_size, 2) if batch_size > 0 else 0
                    ),
                    **extra,
                }
            },
        )

    def log_model_loaded(
        self,
        model_name: str,
        version: str,
        load_time_ms: float,
        **extra: Any,
    ) -> None:
        """Log model load event.

        Args:
            model_name: Name of the loaded model.
            version: Model version.
            load_time_ms: Time to load model in milliseconds.
            **extra: Additional fields to log.
        """
        self.logger.info(
            "Model loaded",
            extra={
                "extra": {
                    "event_type": "model_loaded",
                    "model_name": model_name,
                    "version": version,
                    "load_time_ms": round(load_time_ms, 2),
                    **extra,
                }
            },
        )

    def log_drift_detected(
        self,
        drift_type: str,
        drift_value: float,
        threshold: float,
        **extra: Any,
    ) -> None:
        """Log drift detection event.

        Args:
            drift_type: Type of drift detected.
            drift_value: Measured drift value.
            threshold: Threshold that was exceeded.
            **extra: Additional fields to log.
        """
        self.logger.warning(
            "Model drift detected",
            extra={
                "extra": {
                    "event_type": "drift_detected",
                    "drift_type": drift_type,
                    "drift_value": round(drift_value, 4),
                    "threshold": threshold,
                    **extra,
                }
            },
        )

    def log_error(
        self,
        error_type: str,
        error_message: str,
        **extra: Any,
    ) -> None:
        """Log error event.

        Args:
            error_type: Type of error.
            error_message: Error message.
            **extra: Additional fields to log.
        """
        self.logger.error(
            error_message,
            extra={
                "extra": {
                    "event_type": "error",
                    "error_type": error_type,
                    **extra,
                }
            },
        )
