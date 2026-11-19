"""
Tests for the AQI Predictor API.
Run with:  pytest tests/ -v

NOTE: Integration tests use a MagicMock pipeline because the .pkl was saved
from Google Colab (older scikit-learn) and cannot be unpickled on a different
sklearn version without regenerating it.  The real pipeline is tested end-to-end
inside the Docker container where the sklearn version is pinned.
"""

from contextlib import asynccontextmanager
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

import app.predictor as predictor_module
from app.predictor import PredictionRequest, app, aqi_to_category, request_to_df

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_mock_pipeline(return_value: float = 275.0) -> MagicMock:
    """Return a MagicMock that mimics a fitted sklearn pipeline."""
    mock = MagicMock()
    mock.predict.return_value = [return_value]
    return mock


@pytest.fixture()
def client(monkeypatch):
    """
    TestClient with a mocked pipeline — avoids loading the real .pkl entirely.
    The lifespan's joblib.load is replaced so it returns our MagicMock.
    """
    mock_pipeline = _make_mock_pipeline()

    @asynccontextmanager
    async def _mock_lifespan(application):
        predictor_module._pipeline = mock_pipeline
        yield
        predictor_module._pipeline = None

    with patch.object(app.router, "lifespan_context", _mock_lifespan):
        with TestClient(app) as c:
            yield c


# ---------------------------------------------------------------------------
# Unit tests — pure functions, no HTTP
# ---------------------------------------------------------------------------


class TestAqiCategory:
    def test_good(self):
        assert aqi_to_category(25) == "Good"

    def test_satisfactory(self):
        assert aqi_to_category(75) == "Satisfactory"

    def test_moderate(self):
        assert aqi_to_category(150) == "Moderate"

    def test_poor(self):
        assert aqi_to_category(250) == "Poor"

    def test_very_poor(self):
        assert aqi_to_category(350) == "Very Poor"

    def test_severe(self):
        assert aqi_to_category(450) == "Severe"

    def test_boundary_500(self):
        assert aqi_to_category(500) == "Severe"


class TestRequestToDataFrame:
    def test_shape(self):
        req = PredictionRequest(City="Delhi", **{"PM2.5": 100.0, "PM10": 180.0}, AQI=250.0)
        df = request_to_df(req)
        assert df.shape == (1, 26)

    def test_city_column(self):
        req = PredictionRequest(City="Mumbai")
        df = request_to_df(req)
        assert df["City"].iloc[0] == "Mumbai"

    def test_null_propagation(self):
        req = PredictionRequest(City="Kolkata")
        df = request_to_df(req)
        assert df["PM2.5"].isna().all()


# ---------------------------------------------------------------------------
# API integration tests
# ---------------------------------------------------------------------------

SAMPLE_PAYLOAD = {
    "City": "Delhi",
    "PM2.5": 120.0,
    "PM10": 210.0,
    "NO": 18.5,
    "NO2": 55.0,
    "NOx": 70.0,
    "NH3": 22.0,
    "CO": 1.5,
    "SO2": 14.0,
    "O3": 30.0,
    "Benzene": 3.5,
    "Toluene": 8.2,
    "AQI": 280.0,
    "aqi_lag1": 270.0,
    "aqi_lag2": 265.0,
    "aqi_lag3": 260.0,
    "aqi_delta1": 5.0,
    "aqi_roll7_mean": 271.4,
    "aqi_roll14_mean": 268.0,
    "aqi_roll30_mean": 260.0,
    "aqi_roll7_std": 8.3,
    "month_sin": -0.5,
    "month_cos": 0.866,
    "dayofweek": 2,
    "is_weekend": 0,
    "quarter": 4,
}


class TestHealthEndpoint:
    def test_health_ok(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["model_loaded"] is True


class TestPredictEndpoint:
    def test_predict_returns_200(self, client):
        resp = client.post("/predict", json=SAMPLE_PAYLOAD)
        assert resp.status_code == 200

    def test_predict_response_shape(self, client):
        resp = client.post("/predict", json=SAMPLE_PAYLOAD)
        data = resp.json()
        assert "predicted_next_day_aqi" in data
        assert "aqi_category" in data
        assert "city" in data

    def test_predict_aqi_in_range(self, client):
        resp = client.post("/predict", json=SAMPLE_PAYLOAD)
        aqi = resp.json()["predicted_next_day_aqi"]
        assert 0.0 <= aqi <= 500.0

    def test_predict_with_nulls(self, client):
        """Pipeline imputes missing values — minimal payload should still work."""
        minimal = {"City": "Delhi", "AQI": 200.0}
        resp = client.post("/predict", json=minimal)
        assert resp.status_code == 200

    def test_predict_unknown_city(self, client):
        """OHE with handle_unknown='ignore' handles unseen cities gracefully."""
        payload = {**SAMPLE_PAYLOAD, "City": "UnknownCity"}
        resp = client.post("/predict", json=payload)
        assert resp.status_code == 200

    def test_predict_city_in_response(self, client):
        resp = client.post("/predict", json=SAMPLE_PAYLOAD)
        assert resp.json()["city"] == "Delhi"
