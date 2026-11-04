"""Automated tests for ML Model Integration and Prediction Endpoints (Phase 3)."""

import pytest
import pandas as pd
from pathlib import Path

# pyrefly: ignore[missing-import]
from app.models.database_models import Transaction


@pytest.fixture
def sample_features():
    """Load legitimate sample transaction features."""
    sample_csv = Path(__file__).resolve().parent.parent.parent / "data" / "sample" / "sample_transactions.csv"
    if sample_csv.is_file():
        df = pd.read_csv(sample_csv)
        row = df.iloc[0].to_dict()
        row.pop("Class", None)
        row.pop("class", None)
        return row
    # Fallback to standard 30 features
    features = {f"V{i}": 0.0 for i in range(1, 29)}
    features["Time"] = 0.0
    features["Amount"] = 100.0
    return features


@pytest.fixture
def sample_fraud_features():
    """Load known fraud sample transaction features."""
    sample_csv = Path(__file__).resolve().parent.parent.parent / "data" / "sample" / "sample_transactions.csv"
    if sample_csv.is_file():
        df = pd.read_csv(sample_csv)
        fraud_df = df[df["Class"] == 1]
        if not fraud_df.empty:
            row = fraud_df.iloc[0].to_dict()
            row.pop("Class", None)
            row.pop("class", None)
            return row
    features = {f"V{i}": -2.5 for i in range(1, 29)}
    features["Time"] = 406.0
    features["Amount"] = 999.0
    return features


def test_predict_random_forest_success(client, sample_features):
    """Verify single transaction prediction with Random Forest."""
    payload = {
        "model": "random_forest",
        "transaction": sample_features,
        "threshold": 0.5,
        "already_scaled": True,
    }
    response = client.post("/api/predict", json=payload)
    assert response.status_code == 200
    data = response.get_json()

    # Verify all expected response keys per system specification
    assert "transaction_id" in data
    assert data["model"] == "random_forest"
    assert data["prediction"] in (0, 1)
    assert data["prediction_label"] in ("Legitimate", "Fraud")
    assert 0.0 <= data["fraud_probability"] <= 1.0
    assert 0.0 <= data["risk_score"] <= 100.0
    assert data["risk_level"] in ("Low", "Medium", "High", "Critical")
    assert data["anomaly_flag"] in (0, 1)
    assert isinstance(data["alerts"], list)
    assert "recommendation" in data
    assert data["inference_time_ms"] > 0


def test_predict_logistic_regression_success(client, sample_features):
    """Verify single transaction prediction with Logistic Regression."""
    payload = {
        "model": "logistic_regression",
        "transaction": sample_features,
        "threshold": 0.5,
        "already_scaled": True,
    }
    response = client.post("/api/predict", json=payload)
    assert response.status_code == 200
    data = response.get_json()

    assert data["model"] == "logistic_regression"
    assert data["prediction"] in (0, 1)
    assert 0.0 <= data["fraud_probability"] <= 1.0


def test_predict_missing_feature(client, sample_features):
    """Verify 400 error when required features are missing."""
    incomplete = dict(sample_features)
    incomplete.pop("Amount", None)  # Remove required feature
    incomplete.pop("V14", None)

    payload = {
        "model": "random_forest",
        "transaction": incomplete,
    }
    response = client.post("/api/predict", json=payload)
    assert response.status_code == 400
    data = response.get_json()
    assert data["error"] == "request_error"
    assert "Missing required features" in str(data)


def test_predict_invalid_numeric_value(client, sample_features):
    """Verify 400 error when non-numeric or NaN values are provided."""
    corrupted = dict(sample_features)
    corrupted["Amount"] = "NOT_A_NUMBER"

    payload = {
        "model": "random_forest",
        "transaction": corrupted,
    }
    response = client.post("/api/predict", json=payload)
    assert response.status_code == 400


def test_predict_unavailable_model(client, sample_features):
    """Verify 400 error and no silent substitution when model is not available."""
    payload = {
        "model": "non_existent_deep_model",
        "transaction": sample_features,
    }
    response = client.post("/api/predict", json=payload)
    assert response.status_code == 400
    data = response.get_json()
    assert data["error"] == "model_not_available"
    assert "not available" in data["message"].lower()


def test_prediction_persisted_in_database(client, sample_features, app):
    """Verify that predictions are logged to the transactions SQLite table."""
    payload = {
        "model": "random_forest",
        "transaction": sample_features,
        "already_scaled": True,
        "transaction_id": "TXN-TEST-PERSIST-001",
    }
    response = client.post("/api/predict", json=payload)
    assert response.status_code == 200

    with app.app_context():
        record = Transaction.query.filter_by(transaction_id="TXN-TEST-PERSIST-001").first()
        assert record is not None
        assert record.model_name == "random_forest"
        assert record.risk_level in ("Low", "Medium", "High", "Critical")


def test_get_models_list(client):
    """Verify GET /api/models endpoint returns registered artifacts."""
    response = client.get("/api/models")
    assert response.status_code == 200
    data = response.get_json()
    assert "models" in data
    assert data["total_models"] >= 4


def test_get_models_compare(client):
    """Verify GET /api/models/compare returns authentic comparison data."""
    response = client.get("/api/models/compare")
    assert response.status_code == 200
    data = response.get_json()
    assert "comparison" in data
    assert len(data["comparison"]) >= 4
    rf = next((m for m in data["comparison"] if "random" in m["model_name"].lower()), None)
    assert rf is not None
    assert rf["f1_score"] is not None
    assert rf["f1_score"] > 0.6
