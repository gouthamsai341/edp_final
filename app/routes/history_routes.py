"""
app/routes/history_routes.py

- GET    /api/history            : list previous predictions
- DELETE /api/history/<id>       : delete one history item
- DELETE /api/history            : clear all history
- GET    /history                : HTML history page
"""

from __future__ import annotations

import logging

from flask import Blueprint, jsonify, render_template

from app.services.history_service import (
    clear_all_predictions, delete_prediction, get_all_predictions,
)

logger = logging.getLogger(__name__)
history_bp = Blueprint("history", __name__)


@history_bp.route("/api/history", methods=["GET"])
def api_history():
    try:
        records = get_all_predictions()
        return jsonify({"success": True, "history": [r.to_dict() for r in records]})
    except Exception:  # noqa: BLE001
        logger.exception("Failed to fetch history")
        return jsonify({"success": False, "message": "Could not load prediction history."}), 500


@history_bp.route("/api/history/<int:prediction_id>", methods=["DELETE"])
def api_delete_history_item(prediction_id: int):
    try:
        deleted = delete_prediction(prediction_id)
        if not deleted:
            return jsonify({"success": False, "message": "Record not found."}), 404
        return jsonify({"success": True})
    except Exception:  # noqa: BLE001
        logger.exception("Failed to delete history item %s", prediction_id)
        return jsonify({"success": False, "message": "Could not delete this record."}), 500


@history_bp.route("/api/history", methods=["DELETE"])
def api_clear_history():
    try:
        count = clear_all_predictions()
        return jsonify({"success": True, "deleted": count})
    except Exception:  # noqa: BLE001
        logger.exception("Failed to clear history")
        return jsonify({"success": False, "message": "Could not clear history."}), 500


@history_bp.route("/history", methods=["GET"])
def history_page():
    return render_template("history.html")
