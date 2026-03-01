"""Tests for evaluation metrics and plotting functions."""

import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evaluation.evaluate import (
    evaluate_medical,
    evaluate_model,
    find_optimal_threshold,
    plot_calibration_curve,
    plot_confusion_matrix,
    plot_precision_recall_curve,
    plot_roc_curve,
)
from src.evaluation.metrics import (
    calculate_all_metrics,
    calculate_confusion_matrix_dict,
    calculate_medical_metrics,
    calculate_npv,
    calculate_specificity,
)


class TestMetrics:
    """Tests for custom metrics functions."""

    @pytest.fixture
    def binary_predictions(self):
        """Create sample binary predictions."""
        y_true = np.array([0, 0, 1, 1, 1, 0, 1, 0, 1, 1])
        y_pred = np.array([0, 1, 1, 1, 0, 0, 1, 0, 1, 1])
        y_pred_proba = np.array([0.2, 0.6, 0.8, 0.9, 0.4, 0.3, 0.7, 0.1, 0.85, 0.75])
        return y_true, y_pred, y_pred_proba

    def test_calculate_specificity(self, binary_predictions):
        """Test specificity calculation."""
        y_true, y_pred, _ = binary_predictions
        specificity = calculate_specificity(y_true, y_pred)

        assert 0.0 <= specificity <= 1.0
        # With our data: TN=3, FP=1, so specificity = 3/(3+1) = 0.75
        assert specificity == 0.75

    def test_calculate_npv(self, binary_predictions):
        """Test negative predictive value calculation."""
        y_true, y_pred, _ = binary_predictions
        npv = calculate_npv(y_true, y_pred)

        assert 0.0 <= npv <= 1.0

    def test_calculate_all_metrics(self, binary_predictions):
        """Test calculating all standard metrics."""
        y_true, y_pred, y_pred_proba = binary_predictions
        metrics = calculate_all_metrics(y_true, y_pred, y_pred_proba)

        # Check all expected keys are present
        assert "accuracy" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1" in metrics
        assert "specificity" in metrics
        assert "npv" in metrics
        assert "roc_auc" in metrics
        assert "pr_auc" in metrics

        # Check values are valid
        for key, value in metrics.items():
            assert 0.0 <= value <= 1.0, f"{key} should be between 0 and 1"

    def test_calculate_all_metrics_without_proba(self, binary_predictions):
        """Test calculating metrics without probabilities."""
        y_true, y_pred, _ = binary_predictions
        metrics = calculate_all_metrics(y_true, y_pred)

        # Should not have AUC metrics
        assert "roc_auc" not in metrics
        assert "pr_auc" not in metrics

        # Should have other metrics
        assert "accuracy" in metrics
        assert "precision" in metrics

    def test_calculate_medical_metrics(self, binary_predictions):
        """Test calculating medical-focused metrics."""
        y_true, y_pred, y_pred_proba = binary_predictions
        metrics = calculate_medical_metrics(y_true, y_pred, y_pred_proba)

        # Check expected keys
        assert "sensitivity" in metrics
        assert "specificity" in metrics
        assert "ppv" in metrics
        assert "npv" in metrics
        assert "true_positives" in metrics
        assert "true_negatives" in metrics
        assert "false_positives" in metrics
        assert "false_negatives" in metrics
        assert "total_patients" in metrics
        assert "disease_prevalence" in metrics
        assert "roc_auc" in metrics

        # Check counts add up
        total = (
            metrics["true_positives"]
            + metrics["true_negatives"]
            + metrics["false_positives"]
            + metrics["false_negatives"]
        )
        assert total == metrics["total_patients"]

    def test_calculate_medical_metrics_without_proba(self, binary_predictions):
        """Test medical metrics without probabilities."""
        y_true, y_pred, _ = binary_predictions
        metrics = calculate_medical_metrics(y_true, y_pred)

        assert "roc_auc" not in metrics
        assert "sensitivity" in metrics

    def test_calculate_confusion_matrix_dict(self, binary_predictions):
        """Test confusion matrix as dictionary."""
        y_true, y_pred, _ = binary_predictions
        cm = calculate_confusion_matrix_dict(y_true, y_pred)

        assert "true_positive" in cm
        assert "true_negative" in cm
        assert "false_positive" in cm
        assert "false_negative" in cm

        # All values should be non-negative integers
        for value in cm.values():
            assert isinstance(value, int)
            assert value >= 0


class TestEvaluateModel:
    """Tests for model evaluation functions."""

    @pytest.fixture
    def trained_model(self):
        """Create a trained model with test data."""
        X, y = make_classification(
            n_samples=100,
            n_features=10,
            n_informative=5,
            random_state=42,
        )
        model = LogisticRegression(random_state=42, max_iter=200)
        model.fit(X, y)
        return model, X, y

    def test_evaluate_model(self, trained_model):
        """Test full model evaluation."""
        model, X, y = trained_model
        metrics = evaluate_model(model, X, y)

        # Check expected keys
        assert "accuracy" in metrics
        assert "threshold" in metrics
        assert "confusion_matrix" in metrics
        assert "classification_report" in metrics

        # Check threshold is correct
        assert metrics["threshold"] == 0.5

    def test_evaluate_model_custom_threshold(self, trained_model):
        """Test evaluation with custom threshold."""
        model, X, y = trained_model
        metrics = evaluate_model(model, X, y, threshold=0.3)

        assert metrics["threshold"] == 0.3

    def test_evaluate_medical(self, trained_model):
        """Test medical-focused evaluation."""
        model, X, y = trained_model
        metrics = evaluate_medical(model, X, y)

        assert "sensitivity" in metrics
        assert "specificity" in metrics


class TestPlotFunctions:
    """Tests for plotting functions."""

    @pytest.fixture
    def trained_model(self):
        """Create a trained model with test data."""
        X, y = make_classification(
            n_samples=100,
            n_features=10,
            n_informative=5,
            random_state=42,
        )
        model = LogisticRegression(random_state=42, max_iter=200)
        model.fit(X, y)
        return model, X, y

    @pytest.fixture
    def predictions(self):
        """Create sample predictions."""
        y_true = np.array([0, 0, 1, 1, 1, 0, 1, 0, 1, 1])
        y_pred = np.array([0, 1, 1, 1, 0, 0, 1, 0, 1, 1])
        return y_true, y_pred

    def test_plot_confusion_matrix(self, predictions):
        """Test confusion matrix plotting."""
        y_true, y_pred = predictions
        fig = plot_confusion_matrix(y_true, y_pred)

        assert fig is not None
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_plot_confusion_matrix_with_save(self, predictions):
        """Test saving confusion matrix plot."""
        y_true, y_pred = predictions

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "cm.png"
            fig = plot_confusion_matrix(y_true, y_pred, output_path=output_path)

            assert output_path.exists()
            import matplotlib.pyplot as plt

            plt.close(fig)

    def test_plot_confusion_matrix_custom_labels(self, predictions):
        """Test confusion matrix with custom labels."""
        y_true, y_pred = predictions
        fig = plot_confusion_matrix(y_true, y_pred, labels=["No", "Yes"])

        assert fig is not None
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_plot_roc_curve(self, trained_model):
        """Test ROC curve plotting."""
        model, X, y = trained_model
        fig = plot_roc_curve(model, X, y)

        assert fig is not None
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_plot_roc_curve_with_save(self, trained_model):
        """Test saving ROC curve plot."""
        model, X, y = trained_model

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "roc.png"
            fig = plot_roc_curve(model, X, y, output_path=output_path)

            assert output_path.exists()
            import matplotlib.pyplot as plt

            plt.close(fig)

    def test_plot_precision_recall_curve(self, trained_model):
        """Test precision-recall curve plotting."""
        model, X, y = trained_model
        fig = plot_precision_recall_curve(model, X, y)

        assert fig is not None
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_plot_precision_recall_curve_with_save(self, trained_model):
        """Test saving PR curve plot."""
        model, X, y = trained_model

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "pr.png"
            fig = plot_precision_recall_curve(model, X, y, output_path=output_path)

            assert output_path.exists()
            import matplotlib.pyplot as plt

            plt.close(fig)

    def test_plot_calibration_curve(self, trained_model):
        """Test calibration curve plotting."""
        model, X, y = trained_model
        fig = plot_calibration_curve(model, X, y)

        assert fig is not None
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_plot_calibration_curve_with_save(self, trained_model):
        """Test saving calibration curve plot."""
        model, X, y = trained_model

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "cal.png"
            fig = plot_calibration_curve(model, X, y, output_path=output_path)

            assert output_path.exists()
            import matplotlib.pyplot as plt

            plt.close(fig)


class TestFindOptimalThreshold:
    """Tests for threshold optimization."""

    @pytest.fixture
    def trained_model(self):
        """Create a trained model with validation data."""
        X, y = make_classification(
            n_samples=200,
            n_features=10,
            n_informative=5,
            random_state=42,
        )
        model = LogisticRegression(random_state=42, max_iter=200)
        model.fit(X[:150], y[:150])
        return model, X[150:], y[150:]

    def test_find_optimal_threshold_f1(self, trained_model):
        """Test finding optimal threshold for F1."""
        model, X_val, y_val = trained_model
        threshold = find_optimal_threshold(model, X_val, y_val, optimize_for="f1")

        assert 0.1 <= threshold <= 0.9

    def test_find_optimal_threshold_recall(self, trained_model):
        """Test finding optimal threshold for recall."""
        model, X_val, y_val = trained_model
        threshold = find_optimal_threshold(model, X_val, y_val, optimize_for="recall")

        assert 0.1 <= threshold <= 0.9

    def test_find_optimal_threshold_precision(self, trained_model):
        """Test finding optimal threshold for precision."""
        model, X_val, y_val = trained_model
        threshold = find_optimal_threshold(
            model, X_val, y_val, optimize_for="precision"
        )

        assert 0.1 <= threshold <= 0.9

    def test_find_optimal_threshold_youden(self, trained_model):
        """Test finding optimal threshold using Youden's J."""
        model, X_val, y_val = trained_model
        threshold = find_optimal_threshold(model, X_val, y_val, optimize_for="youden")

        assert 0.1 <= threshold <= 0.9

    def test_find_optimal_threshold_invalid_metric(self, trained_model):
        """Test that invalid metric raises error."""
        model, X_val, y_val = trained_model

        with pytest.raises(ValueError, match="Unknown metric"):
            find_optimal_threshold(model, X_val, y_val, optimize_for="invalid")
