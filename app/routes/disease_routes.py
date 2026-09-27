"""
app/routes/disease_routes.py

- GET /api/diseases             : full disease library
- GET /api/diseases/<class_id>  : one disease's info
- GET /library, /disease/<id>   : HTML pages
- GET /, /about                 : home + about pages
"""

from __future__ import annotations

import logging

from flask import Blueprint, abort, jsonify, render_template

from app.services.disease_service import get_all_diseases, get_disease_by_id

logger = logging.getLogger(__name__)
disease_bp = Blueprint("disease", __name__)


@disease_bp.route("/api/diseases", methods=["GET"])
def api_diseases():
    return jsonify({"success": True, "diseases": get_all_diseases()})


@disease_bp.route("/api/diseases/<path:class_id>", methods=["GET"])
def api_disease_detail(class_id: str):
    info = get_disease_by_id(class_id)
    if info is None:
        return jsonify({"success": False, "message": "Disease not found."}), 404
    return jsonify({"success": True, "id": class_id, **info})


@disease_bp.route("/", methods=["GET"])
def home_page():
    return render_template("index.html")


@disease_bp.route("/about", methods=["GET"])
def about_page():
    return render_template("about.html")


@disease_bp.route("/library", methods=["GET"])
def library_page():
    diseases = get_all_diseases()
    return render_template("library.html", diseases=diseases)


@disease_bp.route("/disease/<path:class_id>", methods=["GET"])
def disease_detail_page(class_id: str):
    info = get_disease_by_id(class_id)
    if info is None:
        abort(404)
    return render_template("disease_detail.html", disease=info, class_id=class_id)
