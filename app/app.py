"""
app/app.py

Flask application factory. Loading the model once here (not per-request)
satisfies the "Performance" requirement, and centralizing error handlers
here satisfies "Error Handling" (never leak stack traces to the client).
"""

from __future__ import annotations

import logging

from flask import Flask, jsonify, render_template

from app.config import AppConfig, ensure_runtime_dirs
from app.extensions import db
from app.services import model_service


def create_app() -> Flask:
    app = Flask(__name__)
    app.config.from_object(AppConfig)

    _configure_logging()
    ensure_runtime_dirs()

    db.init_app(app)
    with app.app_context():
        db.create_all()  # creates instance/database.db + predictions table if missing

    _register_blueprints(app)
    _register_error_handlers(app)

    # Load the ML model once at startup (see app/services/model_service.py).
    # If no trained model exists yet, the app still starts; /api/predict
    # will return a clear, friendly error until a model is trained.
    with app.app_context():
        model_service.warm_up_model()

    return app


def _configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    logging.getLogger("werkzeug").setLevel(logging.WARNING)


def _register_blueprints(app: Flask) -> None:
    from app.routes.prediction_routes import prediction_bp
    from app.routes.history_routes import history_bp
    from app.routes.disease_routes import disease_bp

    app.register_blueprint(prediction_bp)
    app.register_blueprint(history_bp)
    app.register_blueprint(disease_bp)


def _register_error_handlers(app: Flask) -> None:
    logger = logging.getLogger(__name__)

    @app.errorhandler(404)
    def not_found(_e):
        return render_template("404.html"), 404

    @app.errorhandler(413)
    def too_large(_e):
        return jsonify({
            "success": False,
            "status": "error",
            "message": "The uploaded file is too large.",
        }), 413

    @app.errorhandler(500)
    def server_error(e):
        logger.exception("Unhandled server error: %s", e)
        return render_template("500.html"), 500
