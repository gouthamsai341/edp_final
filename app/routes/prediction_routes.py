"""
app/routes/prediction_routes.py

- POST /api/predict          : run a prediction on an uploaded image
- GET  /api/health            : health check
- GET  /detect, /result       : HTML pages for the detection flow
"""

from __future__ import annotations

import logging

from flask import Blueprint, jsonify, render_template, request

from app.services.prediction_service import handle_prediction_request
from app.services import model_service

logger = logging.getLogger(__name__)
prediction_bp = Blueprint("prediction", __name__)


@prediction_bp.route("/api/health", methods=["GET"])
def health():
    return jsonify({
        "status": "healthy",
        "model_ready": model_service.is_model_ready(),
    })


@prediction_bp.route("/api/predict", methods=["POST"])
def predict():
    file_storage = request.files.get("image")
    try:
        response, status_code = handle_prediction_request(file_storage)
    except Exception:  # noqa: BLE001 - never leak a stack trace to the client
        logger.exception("Unexpected error during prediction")
        return jsonify({
            "success": False,
            "status": "error",
            "message": "Something went wrong while analyzing the image. Please try again.",
        }), 500
    return jsonify(response), status_code


@prediction_bp.route("/detect", methods=["GET"])
def detect_page():
    return render_template("detect.html")


@prediction_bp.route("/result", methods=["GET"])
def result_page():
    return render_template("result.html")
