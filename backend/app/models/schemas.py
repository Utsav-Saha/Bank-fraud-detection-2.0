"""Pydantic schemas used for API validation and consistent responses."""

from __future__ import annotations

import math
from typing import Any
from pydantic import BaseModel, ConfigDict, Field, field_validator

REQUIRED_FEATURES: tuple[str, ...] = (
    "Time",
    "V1",
    "V2",
    "V3",
    "V4",
    "V5",
    "V6",
    "V7",
    "V8",
    "V9",
    "V10",
    "V11",
    "V12",
    "V13",
    "V14",
    "V15",
    "V16",
    "V17",
    "V18",
    "V19",
    "V20",
    "V21",
    "V22",
    "V23",
    "V24",
    "V25",
    "V26",
    "V27",
    "V28",
    "Amount",
)

OPTIONAL_IGNORED: frozenset[str] = frozenset({"Class", "class", "transaction_id", "id", "created_at"})


def _finite_number(val: Any) -> float:
    """Validates that a value is numeric and neither NaN nor infinite."""
    try:
        f = float(val)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Value '{val}' is not a valid number") from exc
    if not math.isfinite(f):
        raise ValueError(f"Value '{val}' is not a finite number (NaN or Infinity)")
    return f


class APIError(BaseModel):
    error: str
    message: str
    details: dict[str, Any] | None = None


class ModelAvailability(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    name: str
    filename: str
    available: bool
    loaded: bool = False
    artifact_type: str
    notes: str | None = None


class HealthResponse(BaseModel):
    status: str
    service: str
    environment: str
    database: str
    models_dir: str
    models: list[ModelAvailability]
    missing_required: list[str] = Field(default_factory=list)


class SinglePredictionRequest(BaseModel):
    model: str = "random_forest"
    transaction: dict[str, float]
    already_scaled: bool = False
    threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    transaction_id: str | None = None

    @field_validator("transaction")
    @classmethod
    def check_required_features(cls, v: dict[str, float]) -> dict[str, float]:
        missing = [f for f in REQUIRED_FEATURES if f not in v]
        if missing:
            raise ValueError(f"Missing required features: {missing}")
        for k, val in v.items():
            if k in REQUIRED_FEATURES:
                _finite_number(val)
        return v


class ExplainRequest(BaseModel):
    model: str = "random_forest"
    transaction: dict[str, float]
    already_scaled: bool = False
    top_n: int = Field(default=10, ge=1, le=30)

    @field_validator("transaction")
    @classmethod
    def check_features(cls, v: dict[str, float]) -> dict[str, float]:
        missing = [f for f in REQUIRED_FEATURES if f not in v]
        if missing:
            raise ValueError(f"Missing required features: {missing}")
        for k, val in v.items():
            if k in REQUIRED_FEATURES:
                _finite_number(val)
        return v


class AlertUpdateSchema(BaseModel):
    review_status: str = Field(..., pattern="^(NEW|REVIEWED|INVESTIGATING|DISMISSED|RESOLVED)$")
    notes: str | None = None
