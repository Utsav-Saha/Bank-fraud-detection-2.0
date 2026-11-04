"""Inspect and load trusted local model artifacts with thread-safe caching."""

from __future__ import annotations

import json
import logging
import os
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib

logger = logging.getLogger(__name__)

# Suppress minor scikit-learn unpickling version warnings
try:
    from sklearn.exceptions import InconsistentVersionWarning
    warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
except ImportError:
    pass

KNOWN_MODELS: tuple[dict[str, str], ...] = (
    {
        "name": "random_forest",
        "filename": "random_forest.pkl",
        "artifact_type": "sklearn_joblib",
        "required": "true",
    },
    {
        "name": "logistic_regression",
        "filename": "logistic_regression.pkl",
        "artifact_type": "sklearn_joblib",
        "required": "true",
    },
    {
        "name": "scaler",
        "filename": "scaler.pkl",
        "artifact_type": "sklearn_joblib",
        "required": "true",
    },
    {
        "name": "isolation_forest",
        "filename": "isolation_forest.pkl",
        "artifact_type": "sklearn_joblib",
        "required": "false",
    },
    {
        "name": "ann",
        "filename": "fraud_ann_model.keras",
        "artifact_type": "keras",
        "required": "false",
    },
    {
        "name": "feature_columns",
        "filename": "feature_columns.json",
        "artifact_type": "json",
        "required": "true",
    },
)


class ModelNotAvailableError(RuntimeError):
    """Raised when a requested model is missing or cannot be evaluated."""
    pass


@dataclass(frozen=True)
class ArtifactStatus:
    name: str
    filename: str
    available: bool
    loaded: bool
    artifact_type: str
    required: bool
    notes: str | None
    path: Path

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "filename": self.filename,
            "available": self.available,
            "loaded": self.loaded,
            "artifact_type": self.artifact_type,
            "notes": self.notes,
        }


def inspect_artifacts(models_dir: Path) -> list[ArtifactStatus]:
    registry = get_registry(models_dir)
    results: list[ArtifactStatus] = []
    for spec in KNOWN_MODELS:
        path = models_dir / spec["filename"]
        available = path.is_file()
        is_loaded = False
        notes = None

        if spec["name"] in registry.supervised:
            is_loaded = True
        elif spec["name"] == "scaler" and registry.scaler is not None:
            is_loaded = True
        elif spec["name"] == "isolation_forest" and registry.isolation_forest is not None:
            is_loaded = True
        elif spec["name"] == "ann" and registry.tensorflow_ann is not None:
            is_loaded = True
        elif spec["name"] == "feature_columns" and bool(registry.feature_columns):
            is_loaded = True

        if not available and spec["name"] == "ann":
            notes = "TensorFlow ANN artifact is not present. The API will not substitute another model."
        elif not available and spec["required"] == "true":
            notes = "Required artifact is missing from the trusted models directory."

        results.append(
            ArtifactStatus(
                name=spec["name"],
                filename=spec["filename"],
                available=available,
                loaded=is_loaded,
                artifact_type=spec["artifact_type"],
                required=spec["required"] == "true",
                notes=notes,
                path=path,
            )
        )
    return results


class ModelRegistry:
    """Thread-safe cache of verified in-memory ML/DL models."""

    def __init__(self, models_dir: Path | None = None) -> None:
        self.models_dir = models_dir or Path("models")
        self.supervised: dict[str, Any] = {}
        self.scaler: Any = None
        self.isolation_forest: Any = None
        self.tensorflow_ann: Any = None
        self.feature_columns: list[str] = []
        self._loaded = False

    def load_all(self, models_dir: Path | None = None) -> None:
        if models_dir:
            self.models_dir = models_dir
        if not self.models_dir.is_dir():
            logger.warning("Models directory does not exist: %s", self.models_dir)
            return

        # 1. Feature columns
        fc_path = self.models_dir / "feature_columns.json"
        if fc_path.is_file():
            try:
                with open(fc_path, "r", encoding="utf-8") as f:
                    self.feature_columns = json.load(f)
            except Exception as e:
                logger.error("Failed to load feature_columns.json: %s", e)

        # 2. Scaler
        scaler_path = self.models_dir / "scaler.pkl"
        if scaler_path.is_file():
            try:
                self.scaler = joblib.load(scaler_path)
            except Exception as e:
                logger.error("Failed to load scaler.pkl: %s", e)

        # 3. Random Forest
        rf_path = self.models_dir / "random_forest.pkl"
        if rf_path.is_file():
            try:
                self.supervised["random_forest"] = joblib.load(rf_path)
            except Exception as e:
                logger.error("Failed to load random_forest.pkl: %s", e)

        # 4. Logistic Regression
        lr_path = self.models_dir / "logistic_regression.pkl"
        if lr_path.is_file():
            try:
                self.supervised["logistic_regression"] = joblib.load(lr_path)
            except Exception as e:
                logger.error("Failed to load logistic_regression.pkl: %s", e)

        # 5. Isolation Forest
        iso_path = self.models_dir / "isolation_forest.pkl"
        if iso_path.is_file():
            try:
                self.isolation_forest = joblib.load(iso_path)
            except Exception as e:
                logger.error("Failed to load isolation_forest.pkl: %s", e)

        # 6. Optional TensorFlow ANN
        ann_path = self.models_dir / "fraud_ann_model.keras"
        if ann_path.is_file():
            try:
                import tensorflow as tf
                self.tensorflow_ann = tf.keras.models.load_model(ann_path)
                self.supervised["ann"] = self.tensorflow_ann
                self.supervised["tensorflow_ann"] = self.tensorflow_ann
            except Exception as e:
                logger.error("Failed to load fraud_ann_model.keras: %s", e)

        self._loaded = True

    def require(self, model_name: str) -> Any:
        normalized = model_name.strip().lower()
        if normalized in {"ann", "tensorflow_ann", "keras"}:
            if self.tensorflow_ann is not None:
                return self.tensorflow_ann
            raise ModelNotAvailableError(
                "TensorFlow ANN model ('fraud_ann_model.keras') is not installed or available on this system. "
                "Per system policy, another model will not be silently substituted."
            )

        if normalized in self.supervised:
            return self.supervised[normalized]

        available = list(self.supervised.keys())
        raise ModelNotAvailableError(
            f"Requested model '{model_name}' is not available. Available models: {available}"
        )


_REGISTRY: ModelRegistry | None = None


def get_registry(models_dir: Path | None = None) -> ModelRegistry:
    global _REGISTRY
    if _REGISTRY is None:
        _REGISTRY = ModelRegistry(models_dir)
        _REGISTRY.load_all(models_dir)
    elif models_dir and not _REGISTRY._loaded:
        _REGISTRY.load_all(models_dir)
    return _REGISTRY
