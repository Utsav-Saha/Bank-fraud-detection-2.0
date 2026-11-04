"""Flask Application Factory for SecureBank AI Fraud Detection System."""

from __future__ import annotations

import logging
import os
from pathlib import Path

from flask import Flask, jsonify
from werkzeug.exceptions import HTTPException

# pyrefly: ignore[missing-import]
from app.config import get_config
# pyrefly: ignore[missing-import]
from app.extensions import cors, db, migrate
# pyrefly: ignore[missing-import]
from app.utils.metrics_seed import seed_model_metadata_from_csv
# pyrefly: ignore[missing-import]
from app.utils.model_loader import get_registry

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [%(name)s]: %(message)s",
)
logger = logging.getLogger("fraud_detection_api")


def create_app(config_name: str | None = None) -> Flask:
    """Application factory for the fraud intelligence API."""
    app = Flask(__name__)
    config_class = get_config(config_name)
    app.config.from_object(config_class)

    # Initialize extensions
    db.init_app(app)
    migrate.init_app(app, db)
    cors.init_app(
        app,
        resources={r"/api/*": {"origins": app.config.get("CORS_ORIGINS", "*")}},
        supports_credentials=True,
    )

    # Register blueprints
    _register_blueprints(app)

    # Register error handlers
    _register_error_handlers(app)

    # Initialize database tables and preload model cache
    with app.app_context():
        try:
            db.create_all()
            metrics_dir = app.config.get("METRICS_DIR")
            if metrics_dir and Path(metrics_dir).is_dir():
                seeded = seed_model_metadata_from_csv(Path(metrics_dir))
                if seeded:
                    logger.info("Seeded %d model metadata rows from %s", seeded, metrics_dir)
        except Exception as e:
            logger.warning("Database initialization notice: %s", e)

        # Pre-warm model registry in background
        models_dir = app.config.get("MODELS_DIR")
        if models_dir and Path(models_dir).is_dir():
            try:
                registry = get_registry(Path(models_dir))
                logger.info(
                    "Model registry initialized. Available models: %s",
                    list(registry.supervised.keys()),
                )
            except Exception as e:
                logger.warning("Could not pre-warm model registry: %s", e)

    return app


def _register_blueprints(app: Flask) -> None:
    """Dynamically register available blueprints under /api prefix."""
    # pyrefly: ignore[missing-import]
    from app.routes.health_routes import health_bp
    app.register_blueprint(health_bp, url_prefix="/api")

    # Conditionally register other route blueprints if already present
    try:
        # pyrefly: ignore[missing-import]
        from app.routes.prediction_routes import prediction_bp
        app.register_blueprint(prediction_bp, url_prefix="/api")
    except ImportError:
        pass

    try:
        # pyrefly: ignore[missing-import]
        from app.routes.analytics_routes import analytics_bp
        app.register_blueprint(analytics_bp, url_prefix="/api")
    except ImportError:
        pass

    try:
        # pyrefly: ignore[missing-import]
        from app.routes.alert_routes import alert_bp
        app.register_blueprint(alert_bp, url_prefix="/api")
    except ImportError:
        pass

    try:
        # pyrefly: ignore[missing-import]
        from app.routes.model_routes import model_bp
        app.register_blueprint(model_bp, url_prefix="/api")
    except ImportError:
        pass

    try:
        # pyrefly: ignore[missing-import]
        from app.routes.investigation_routes import investigation_bp
        app.register_blueprint(investigation_bp, url_prefix="/api")
    except ImportError:
        pass


def _register_error_handlers(app: Flask) -> None:
    """Consistent JSON error responses matching APIError schema."""

    @app.errorhandler(400)
    def bad_request(error):
        msg = getattr(error, "description", "Bad Request")
        return jsonify({"error": "bad_request", "message": str(msg)}), 400

    @app.errorhandler(404)
    def not_found(error):
        return jsonify({"error": "not_found", "message": "The requested API endpoint was not found."}), 404

    @app.errorhandler(405)
    def method_not_allowed(error):
        return jsonify({"error": "method_not_allowed", "message": "HTTP method not allowed for this route."}), 405

    @app.errorhandler(413)
    def payload_too_large(error):
        max_bytes = app.config.get("MAX_CONTENT_LENGTH", 10 * 1024 * 1024)
        return jsonify({
            "error": "payload_too_large",
            "message": f"Uploaded file exceeds the maximum allowed size of {max_bytes} bytes.",
        }), 413

    @app.errorhandler(HTTPException)
    def handle_http_exception(error):
        return jsonify({
            "error": "http_error",
            "message": error.description,
            "code": error.code,
        }), error.code or 500

    @app.errorhandler(Exception)
    def handle_unhandled_exception(error):
        logger.exception("Unhandled server exception: %s", error)
        # Never expose internal stack traces to clients in production
        return jsonify({
            "error": "internal_server_error",
            "message": "An unexpected server error occurred while processing the request.",
        }), 500
