"""Tests for monitoring utilities."""

import logging
from datetime import datetime, timedelta

from src.monitoring.model_monitor import ModelPerformanceMonitor
from src.monitoring.prediction_logger import JSONFormatter, PredictionLogger


class TestModelPerformanceMonitor:
    """Tests for ModelPerformanceMonitor."""

    def test_init(self):
        """Monitor should initialize with empty state."""
        monitor = ModelPerformanceMonitor(window_size=100)
        assert len(monitor.predictions) == 0
        assert len(monitor.probabilities) == 0
        assert monitor.window_size == 100

    def test_record_prediction(self):
        """Should record predictions correctly."""
        monitor = ModelPerformanceMonitor()
        monitor.record_prediction(1, 0.75)
        monitor.record_prediction(0, 0.25)

        assert len(monitor.predictions) == 2
        assert list(monitor.predictions) == [1, 0]
        assert list(monitor.probabilities) == [0.75, 0.25]

    def test_window_size_limit(self):
        """Should respect window size limit."""
        monitor = ModelPerformanceMonitor(window_size=3)

        for i in range(5):
            monitor.record_prediction(1, 0.5)

        assert len(monitor.predictions) == 3

    def test_get_statistics_empty(self):
        """Should return error for empty predictions."""
        monitor = ModelPerformanceMonitor()
        stats = monitor.get_statistics()
        assert "error" in stats

    def test_get_statistics(self):
        """Should calculate correct statistics."""
        monitor = ModelPerformanceMonitor()
        monitor.record_prediction(1, 0.8)
        monitor.record_prediction(0, 0.2)
        monitor.record_prediction(1, 0.6)
        monitor.record_prediction(0, 0.4)

        stats = monitor.get_statistics()

        assert stats["total_predictions"] == 4
        assert stats["positive_rate"] == 0.5
        assert stats["mean_probability"] == 0.5
        assert stats["min_probability"] == 0.2
        assert stats["max_probability"] == 0.8

    def test_set_baseline(self):
        """Should set baseline statistics."""
        monitor = ModelPerformanceMonitor()
        monitor.set_baseline(mean_prob=0.5, std_prob=0.2, positive_rate=0.4)

        assert monitor.baseline_mean_prob == 0.5
        assert monitor.baseline_std_prob == 0.2
        assert monitor.baseline_positive_rate == 0.4

    def test_check_drift_no_baseline(self):
        """Should return error if baseline not set."""
        monitor = ModelPerformanceMonitor()
        monitor.record_prediction(1, 0.8)
        result = monitor.check_drift()
        assert "error" in result

    def test_check_drift_no_drift(self):
        """Should not detect drift when distribution is similar."""
        monitor = ModelPerformanceMonitor()
        monitor.set_baseline(mean_prob=0.5, std_prob=0.2, positive_rate=0.5)

        # Add predictions similar to baseline
        for _ in range(10):
            monitor.record_prediction(1, 0.55)
            monitor.record_prediction(0, 0.45)

        result = monitor.check_drift()
        assert not result["drift_detected"]

    def test_check_drift_detected(self):
        """Should detect drift when distribution changes significantly."""
        monitor = ModelPerformanceMonitor()
        monitor.set_baseline(mean_prob=0.3, std_prob=0.1, positive_rate=0.3)

        # Add predictions very different from baseline
        for _ in range(10):
            monitor.record_prediction(1, 0.9)

        result = monitor.check_drift()
        assert result["drift_detected"]
        assert result["probability_drift_zscore"] > 2.0

    def test_get_recent_summary(self):
        """Should summarize recent predictions."""
        monitor = ModelPerformanceMonitor()

        # Add old predictions
        old_time = datetime.utcnow() - timedelta(hours=2)
        monitor.record_prediction(0, 0.2, timestamp=old_time)

        # Add recent predictions
        now = datetime.utcnow()
        monitor.record_prediction(1, 0.8, timestamp=now)
        monitor.record_prediction(1, 0.75, timestamp=now)

        summary = monitor.get_recent_summary(minutes=60)

        assert summary["prediction_count"] == 2
        assert summary["high_risk_count"] == 2
        assert summary["positive_rate"] == 1.0

    def test_reset(self):
        """Should clear all predictions."""
        monitor = ModelPerformanceMonitor()
        monitor.record_prediction(1, 0.8)
        monitor.record_prediction(0, 0.2)
        monitor.reset()

        assert len(monitor.predictions) == 0
        assert len(monitor.probabilities) == 0


class TestPredictionLogger:
    """Tests for PredictionLogger."""

    def test_init_default(self):
        """Should initialize with default logger."""
        logger = PredictionLogger()
        assert logger.logger is not None

    def test_init_custom_logger(self):
        """Should accept custom logger."""
        custom_logger = logging.getLogger("test")
        pred_logger = PredictionLogger(logger=custom_logger)
        assert pred_logger.logger == custom_logger

    def test_log_prediction(self, caplog):
        """Should log prediction events."""
        logger = PredictionLogger(json_format=False)

        with caplog.at_level(logging.INFO):
            logger.log_prediction(
                patient_id="test-123",
                prediction=1,
                probability=0.85,
                risk_level="high",
                latency_ms=15.5,
            )

        assert "Prediction made" in caplog.text

    def test_log_batch_prediction(self, caplog):
        """Should log batch prediction events."""
        logger = PredictionLogger(json_format=False)

        with caplog.at_level(logging.INFO):
            logger.log_batch_prediction(
                batch_size=10,
                high_risk_count=3,
                total_latency_ms=150.0,
            )

        assert "Batch prediction completed" in caplog.text

    def test_log_model_loaded(self, caplog):
        """Should log model load events."""
        logger = PredictionLogger(json_format=False)

        with caplog.at_level(logging.INFO):
            logger.log_model_loaded(
                model_name="logistic_regression",
                version="1.0.0",
                load_time_ms=500.0,
            )

        assert "Model loaded" in caplog.text

    def test_log_drift_detected(self, caplog):
        """Should log drift detection events."""
        logger = PredictionLogger(json_format=False)

        with caplog.at_level(logging.WARNING):
            logger.log_drift_detected(
                drift_type="probability",
                drift_value=3.5,
                threshold=2.0,
            )

        assert "drift detected" in caplog.text

    def test_log_error(self, caplog):
        """Should log error events."""
        logger = PredictionLogger(json_format=False)

        with caplog.at_level(logging.ERROR):
            logger.log_error(
                error_type="prediction_error",
                error_message="Model failed to predict",
            )

        assert "Model failed to predict" in caplog.text


class TestJSONFormatter:
    """Tests for JSONFormatter."""

    def test_format_basic(self):
        """Should format log record as JSON."""
        formatter = JSONFormatter()
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=10,
            msg="Test message",
            args=(),
            exc_info=None,
        )

        output = formatter.format(record)

        import json

        data = json.loads(output)
        assert data["message"] == "Test message"
        assert data["level"] == "INFO"
        assert "timestamp" in data

    def test_format_with_extra(self):
        """Should include extra fields in output."""
        formatter = JSONFormatter()
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=10,
            msg="Test message",
            args=(),
            exc_info=None,
        )
        record.extra = {"custom_field": "custom_value"}

        output = formatter.format(record)

        import json

        data = json.loads(output)
        assert data["custom_field"] == "custom_value"
