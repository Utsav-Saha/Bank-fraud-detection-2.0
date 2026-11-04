"""Small shared helpers."""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any

from flask import jsonify


def make_transaction_id(prefix: str = "TXN") -> str:
    """Generate a unique human-readable transaction identifier."""
    return f"{prefix}-{uuid.uuid4().hex[:8].upper()}"


def json_error(message: str, status: int, error: str = "request_error", details: dict[str, Any] | None = None):
    payload = {"error": error, "message": message}
    if details:
        payload["details"] = details
    return jsonify(payload), status


def path_exists(path: Path) -> bool:
    try:
        return path.is_file()
    except OSError:
        return False
