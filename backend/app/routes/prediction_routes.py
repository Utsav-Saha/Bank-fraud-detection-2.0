"""Prediction routes for single and batch fraud scoring."""

from __future__ import annotations

import logging
from flask import Blueprint, jsonify, request
from pydantic import ValidationError

# pyrefly: ignore[missing-import]
from app.models.schemas import SinglePredictionRequest
# pyrefly: ignore[missing-import]
from app.services.alert_service import persist_prediction
# pyrefly: ignore[missing-import]
from app.services.prediction_service import predict_transaction
# pyrefly: ignore[missing-import]
from app.utils.helpers import json_error
# pyrefly: ignore[missing-import]
from app.utils.model_loader import ModelNotAvailableError

logger = logging.getLogger(__name__)

prediction_bp = Blueprint("prediction", __name__)


@prediction_bp.post("/predict")
def predict_single():
    """Score a single credit card transaction for fraud risk and anomalies.
    
    Expected JSON payload:
    {
      "model": "random_forest",  // "random_forest" | "logistic_regression" | "ann"
      "transaction": { "Time": 0.0, "V1": -1.35, ..., "Amount": 149.99 },
      "threshold": 0.5,           // optional classification threshold
      "already_scaled": false     // optional boolean flag
    }
    """
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return json_error("Invalid JSON body. Expected a JSON object.", status=400)

    try:
        validated = SinglePredictionRequest(**data)
    except ValidationError as err:
        clean_errors = [
            {
                "field": ".".join(str(item) for item in e.get("loc", [])),
                "message": str(e.get("msg", "")),
                "type": str(e.get("type", "")),
            }
            for e in err.errors()
        ]
        return json_error(
            "Transaction validation failed.",
            status=400,
            details={"errors": clean_errors},
        )
    except Exception as exc:
        return json_error(f"Validation error: {exc}", status=400)

    try:
        result = predict_transaction(
            validated.transaction,
            model_name=validated.model,
            already_scaled=validated.already_scaled,
            threshold=validated.threshold,
            transaction_id=validated.transaction_id,
        )
        
        # Persist prediction and triggered rules to database
        try:
            persist_prediction(result)
        except Exception as db_exc:
            logger.warning("Could not persist transaction %s to database: %s", result.get("transaction_id"), db_exc)

        return jsonify(result), 200

    except ModelNotAvailableError as model_err:
        return json_error(str(model_err), status=400, error="model_not_available")
    except Exception as exc:
        logger.exception("Error during transaction prediction: %s", exc)
        return json_error(f"Inference error: {exc}", status=500, error="inference_error")
