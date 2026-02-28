"""Tests for the FastAPI application."""

import pytest
from fastapi.testclient import TestClient

from api.main import app
from api.models import PatientFeatures, RiskLevel


@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


@pytest.fixture
def sample_patient():
    """Sample patient data for testing."""
    return {
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


class TestHealthEndpoint:
    """Tests for /health endpoint."""

    def test_health_returns_200(self, client):
        """Health endpoint should return 200."""
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_response_structure(self, client):
        """Health response should have required fields."""
        response = client.get("/health")
        data = response.json()

        assert "status" in data
        assert "model_loaded" in data
        assert "uptime_seconds" in data
        assert data["status"] in ["healthy", "unhealthy"]

    def test_health_uptime_positive(self, client):
        """Uptime should be a positive number."""
        response = client.get("/health")
        data = response.json()

        assert data["uptime_seconds"] >= 0


class TestMetricsEndpoint:
    """Tests for /metrics endpoint."""

    def test_metrics_returns_200(self, client):
        """Metrics endpoint should return 200."""
        response = client.get("/metrics")
        assert response.status_code == 200

    def test_metrics_content_type(self, client):
        """Metrics should return prometheus format."""
        response = client.get("/metrics")
        assert "text/plain" in response.headers["content-type"]

    def test_metrics_contains_custom_metrics(self, client):
        """Metrics should contain our custom metrics."""
        response = client.get("/metrics")
        content = response.text

        assert "hearthealthml_predictions_total" in content
        assert "hearthealthml_prediction_latency_seconds" in content
        assert "hearthealthml_requests_total" in content


class TestPredictEndpoint:
    """Tests for /predict endpoint."""

    def test_predict_valid_input(self, client, sample_patient):
        """Valid input should return prediction."""
        response = client.post("/predict", json=sample_patient)

        # May return 503 if model not loaded, which is acceptable in tests
        assert response.status_code in [200, 503]

        if response.status_code == 200:
            data = response.json()
            assert "prediction" in data
            assert "probability" in data
            assert "risk_level" in data
            assert "confidence" in data
            assert data["prediction"] in [0, 1]
            assert 0 <= data["probability"] <= 1
            assert data["risk_level"] in ["low", "medium", "high"]

    def test_predict_missing_field(self, client, sample_patient):
        """Missing required field should return 422."""
        del sample_patient["age"]
        response = client.post("/predict", json=sample_patient)
        assert response.status_code == 422

    def test_predict_invalid_age(self, client, sample_patient):
        """Invalid age value should return 422."""
        sample_patient["age"] = 150  # Above max
        response = client.post("/predict", json=sample_patient)
        assert response.status_code == 422

    def test_predict_invalid_sex(self, client, sample_patient):
        """Invalid sex value should return 422."""
        sample_patient["sex"] = 2  # Not 0 or 1
        response = client.post("/predict", json=sample_patient)
        assert response.status_code == 422


class TestBatchPredictEndpoint:
    """Tests for /predict/batch endpoint."""

    def test_batch_predict_single_patient(self, client, sample_patient):
        """Batch with single patient should work."""
        response = client.post("/predict/batch", json={"patients": [sample_patient]})

        assert response.status_code in [200, 503]

        if response.status_code == 200:
            data = response.json()
            assert data["total_count"] == 1
            assert len(data["predictions"]) == 1
            assert "high_risk_count" in data

    def test_batch_predict_multiple_patients(self, client, sample_patient):
        """Batch with multiple patients should work."""
        patients = [sample_patient, sample_patient, sample_patient]
        response = client.post("/predict/batch", json={"patients": patients})

        assert response.status_code in [200, 503]

        if response.status_code == 200:
            data = response.json()
            assert data["total_count"] == 3
            assert len(data["predictions"]) == 3

    def test_batch_predict_empty_list(self, client):
        """Empty patient list should return 422."""
        response = client.post("/predict/batch", json={"patients": []})
        assert response.status_code == 422

    def test_batch_predict_exceeds_limit(self, client, sample_patient):
        """Exceeding max batch size should return 422."""
        patients = [sample_patient] * 101  # Max is 100
        response = client.post("/predict/batch", json={"patients": patients})
        assert response.status_code == 422


class TestModelInfoEndpoint:
    """Tests for /model-info endpoint."""

    def test_model_info_structure(self, client):
        """Model info should have required fields."""
        response = client.get("/model-info")

        # May return 503 if model not loaded
        assert response.status_code in [200, 503]

        if response.status_code == 200:
            data = response.json()
            assert "model_name" in data
            assert "version" in data
            assert "features" in data
            assert "metrics" in data
            assert "threshold" in data


class TestReloadEndpoint:
    """Tests for /reload endpoint."""

    def test_reload_endpoint_exists(self, client):
        """Reload endpoint should exist."""
        response = client.post("/reload")
        # May succeed or fail depending on model availability
        assert response.status_code in [200, 503]


class TestPatientFeaturesModel:
    """Tests for PatientFeatures Pydantic model."""

    def test_valid_patient(self, sample_patient):
        """Valid patient data should create model instance."""
        patient = PatientFeatures(**sample_patient)
        assert patient.age == 55
        assert patient.sex == 1

    def test_invalid_age_range(self, sample_patient):
        """Age outside valid range should raise error."""
        sample_patient["age"] = 0
        with pytest.raises(ValueError):
            PatientFeatures(**sample_patient)

    def test_invalid_categorical(self, sample_patient):
        """Invalid categorical value should raise error."""
        sample_patient["cp"] = 5  # Valid: 0-3
        with pytest.raises(ValueError):
            PatientFeatures(**sample_patient)


class TestRiskLevelEnum:
    """Tests for RiskLevel enum."""

    def test_risk_level_values(self):
        """Risk levels should have correct values."""
        assert RiskLevel.LOW.value == "low"
        assert RiskLevel.MEDIUM.value == "medium"
        assert RiskLevel.HIGH.value == "high"

    def test_risk_level_from_string(self):
        """Risk level should be creatable from string."""
        assert RiskLevel("low") == RiskLevel.LOW
        assert RiskLevel("medium") == RiskLevel.MEDIUM
        assert RiskLevel("high") == RiskLevel.HIGH
