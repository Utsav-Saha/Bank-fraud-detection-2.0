"""Automated tests for Backend API Foundation (Phase 2)."""

import pytest
# pyrefly: ignore[missing-import]
from app.models.database_models import ModelMetadata


def test_health_check_endpoint(client):
    """Verify GET /api/health returns 200 OK and valid JSON structure."""
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.get_json()

    assert data["status"] == "ok"
    assert data["service"] == "banking-fraud-detection-api"
    assert data["database"] == "connected"
    assert isinstance(data["models"], list)
    assert len(data["models"]) >= 4

    # Ensure required models are marked available
    model_names = {m["name"]: m for m in data["models"]}
    assert "random_forest" in model_names
    assert model_names["random_forest"]["available"] is True
    assert model_names["random_forest"]["loaded"] is True

    assert "logistic_regression" in model_names
    assert model_names["logistic_regression"]["available"] is True

    assert "scaler" in model_names
    assert model_names["scaler"]["available"] is True


def test_404_structured_json_error(client):
    """Verify 404 returns consistent JSON matching APIError schema."""
    response = client.get("/api/non_existent_route")
    assert response.status_code == 404
    data = response.get_json()
    assert data["error"] == "not_found"
    assert "message" in data


def test_405_structured_json_error(client):
    """Verify 405 returns consistent JSON matching APIError schema."""
    response = client.post("/api/health")  # Health is GET only
    assert response.status_code == 405
    data = response.get_json()
    assert data["error"] == "method_not_allowed"


def test_model_metadata_seeded(app):
    """Verify that ModelMetadata table is seeded from existing CSV."""
    with app.app_context():
        records = ModelMetadata.query.all()
        # Should have seeded Random Forest, Logistic Regression, Decision Tree, KNN
        assert len(records) >= 4
        rf_meta = ModelMetadata.query.filter_by(model_name="Random Forest").first()
        assert rf_meta is not None
        assert rf_meta.roc_auc is not None
        assert rf_meta.roc_auc > 0.90
