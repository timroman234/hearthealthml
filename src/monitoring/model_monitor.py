"""Model performance monitoring for drift detection and statistics."""

from collections import deque
from datetime import datetime, timedelta
from typing import Any

import numpy as np


class ModelPerformanceMonitor:
    """Monitor model predictions and detect potential issues.

    Tracks predictions over a sliding window and compares against
    baseline statistics to detect distribution drift.

    Attributes:
        window_size: Maximum number of predictions to track.
        predictions: Deque of recent predictions.
        probabilities: Deque of recent prediction probabilities.
        timestamps: Deque of prediction timestamps.
    """

    def __init__(self, window_size: int = 1000):
        """Initialize the monitor.

        Args:
            window_size: Number of predictions to keep in sliding window.
        """
        self.window_size = window_size
        self.predictions: deque = deque(maxlen=window_size)
        self.probabilities: deque = deque(maxlen=window_size)
        self.timestamps: deque = deque(maxlen=window_size)

        # Baseline statistics (set during initialization)
        self.baseline_mean_prob: float | None = None
        self.baseline_std_prob: float | None = None
        self.baseline_positive_rate: float | None = None

    def set_baseline(
        self,
        mean_prob: float,
        std_prob: float,
        positive_rate: float,
    ) -> None:
        """Set baseline statistics from training/validation data.

        Args:
            mean_prob: Mean prediction probability from baseline.
            std_prob: Standard deviation of probabilities.
            positive_rate: Fraction of positive predictions in baseline.
        """
        self.baseline_mean_prob = mean_prob
        self.baseline_std_prob = std_prob
        self.baseline_positive_rate = positive_rate

    def record_prediction(
        self,
        prediction: int,
        probability: float,
        timestamp: datetime | None = None,
    ) -> None:
        """Record a new prediction.

        Args:
            prediction: Binary prediction (0 or 1).
            probability: Prediction probability for positive class.
            timestamp: Timestamp of prediction (defaults to now).
        """
        self.predictions.append(prediction)
        self.probabilities.append(probability)
        self.timestamps.append(timestamp or datetime.utcnow())

    def get_statistics(self) -> dict[str, Any]:
        """Get current prediction statistics.

        Returns:
            Dictionary with prediction statistics.
        """
        if len(self.predictions) == 0:
            return {"error": "No predictions recorded"}

        probs = np.array(self.probabilities)
        preds = np.array(self.predictions)

        return {
            "total_predictions": len(self.predictions),
            "mean_probability": float(np.mean(probs)),
            "std_probability": float(np.std(probs)),
            "positive_rate": float(np.mean(preds)),
            "min_probability": float(np.min(probs)),
            "max_probability": float(np.max(probs)),
            "high_risk_count": int(np.sum(probs >= 0.7)),
            "medium_risk_count": int(np.sum((probs >= 0.3) & (probs < 0.7))),
            "low_risk_count": int(np.sum(probs < 0.3)),
        }

    def check_drift(self, threshold: float = 2.0) -> dict[str, Any]:
        """Check for prediction distribution drift.

        Compares current prediction distribution against baseline
        using z-score for probability mean and absolute difference
        for positive rate.

        Args:
            threshold: Z-score threshold for drift detection.

        Returns:
            Dictionary with drift analysis results.
        """
        if self.baseline_mean_prob is None:
            return {"error": "Baseline not set"}

        stats = self.get_statistics()
        if "error" in stats:
            return stats

        # Calculate z-scores for drift detection
        if self.baseline_std_prob is not None and self.baseline_std_prob > 0:
            prob_drift = (
                abs(stats["mean_probability"] - self.baseline_mean_prob)
                / self.baseline_std_prob
            )
        else:
            prob_drift = 0.0

        rate_drift = abs(stats["positive_rate"] - (self.baseline_positive_rate or 0.0))

        drift_detected = prob_drift > threshold or rate_drift > 0.1

        return {
            "probability_drift_zscore": float(prob_drift),
            "positive_rate_drift": float(rate_drift),
            "drift_detected": drift_detected,
            "current_mean_prob": stats["mean_probability"],
            "baseline_mean_prob": self.baseline_mean_prob,
            "current_positive_rate": stats["positive_rate"],
            "baseline_positive_rate": self.baseline_positive_rate,
            "recommendation": (
                "Investigate model performance - consider retraining"
                if drift_detected
                else "No action needed"
            ),
        }

    def get_recent_summary(self, minutes: int = 60) -> dict[str, Any]:
        """Get summary of recent predictions.

        Args:
            minutes: Number of minutes to look back.

        Returns:
            Dictionary with recent prediction summary.
        """
        cutoff = datetime.utcnow() - timedelta(minutes=minutes)

        recent_probs = []
        recent_preds = []

        for ts, prob, pred in zip(
            self.timestamps, self.probabilities, self.predictions
        ):
            if ts >= cutoff:
                recent_probs.append(prob)
                recent_preds.append(pred)

        if len(recent_probs) == 0:
            return {"error": f"No predictions in last {minutes} minutes"}

        return {
            "period_minutes": minutes,
            "prediction_count": len(recent_probs),
            "mean_probability": float(np.mean(recent_probs)),
            "high_risk_count": sum(1 for p in recent_probs if p >= 0.7),
            "medium_risk_count": sum(1 for p in recent_probs if 0.3 <= p < 0.7),
            "low_risk_count": sum(1 for p in recent_probs if p < 0.3),
            "positive_rate": float(np.mean(recent_preds)),
        }

    def reset(self) -> None:
        """Clear all recorded predictions."""
        self.predictions.clear()
        self.probabilities.clear()
        self.timestamps.clear()
