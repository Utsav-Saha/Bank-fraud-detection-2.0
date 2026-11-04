"""Model inventory and comparative evaluation routes."""

from __future__ import annotations

from pathlib import Path
from flask import Blueprint, current_app, jsonify
import pandas as pd

from app.models.database_models import ModelMetadata
from app.utils.model_loader import inspect_artifacts

model_bp = Blueprint("models", __name__)


@model_bp.get("/models")
def list_models():
    """Return status and metadata of all registered ML/DL models."""
    models_dir = current_app.config["MODELS_DIR"]
    artifacts = inspect_artifacts(models_dir)
    return jsonify({
        "models_dir": str(models_dir),
        "total_models": len(artifacts),
        "models": [item.as_dict() for item in artifacts],
    }), 200


@model_bp.get("/models/compare")
def compare_models():
    """Return authentic evaluated model performance metrics."""
    records = ModelMetadata.query.all()
    models_list = []

    if records:
        for r in records:
            models_list.append({
                "model_name": r.model_name,
                "model_version": r.model_version,
                "accuracy": r.accuracy,
                "precision": r.precision,
                "recall": r.recall,
                "f1_score": r.f1_score,
                "roc_auc": r.roc_auc,
                "pr_auc": r.pr_auc,
                "training_time_sec": r.training_time_sec,
                "source_file": r.source_file,
            })
    else:
        # Fallback to reading CSV if database was not yet seeded
        csv_path = current_app.config.get("METRICS_CSV")
        if csv_path and Path(csv_path).is_file():
            df = pd.read_csv(csv_path)
            for _, row in df.iterrows():
                models_list.append({
                    "model_name": row.get("Model"),
                    "model_version": "notebook-ml-algo",
                    "accuracy": float(row.get("Accuracy", 0.0)),
                    "precision": float(row.get("Precision", 0.0)),
                    "recall": float(row.get("Recall", 0.0)),
                    "f1_score": float(row.get("F1 Score", 0.0)),
                    "roc_auc": float(row.get("ROC-AUC", 0.0)),
                    "training_time_sec": float(row.get("Training Time (sec)", 0.0)),
                })

    return jsonify({
        "comparison": models_list,
        "evaluation_notes": (
            "Because credit card fraud data is highly imbalanced (fraud rate < 0.2%), "
            "evaluations prioritize Precision, Recall, F1-Score, and PR-AUC/ROC-AUC over raw accuracy."
        ),
        "primary_metric": "F1-Score / Recall",
        "recommended_production_model": "Random Forest",
    }), 200
