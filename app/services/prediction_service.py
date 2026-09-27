"""
app/services/prediction_service.py

Orchestrates a full prediction request: validate upload -> save file
securely -> run model inference -> persist to history -> return a
consistent response dict for the API/routes layer.
"""

from __future__ import annotations

import logging

from app.config import AppConfig
from app.services import model_service
from app.services.history_service import save_prediction_record
from app.utils.image_validation import (
    UploadValidationError, generate_safe_filename, validate_upload,
)

logger = logging.getLogger(__name__)


def handle_prediction_request(file_storage) -> tuple[dict, int]:
    """
    Returns (response_dict, http_status_code).
    Never raises - all failure modes are converted into a friendly,
    consistent JSON-able response so routes stay simple.
    """
    try:
        validate_upload(file_storage)
    except UploadValidationError as exc:
        logger.info("Upload validation failed: %s", exc)
        return {"success": False, "status": "error", "message": str(exc)}, 400

    safe_filename = generate_safe_filename(file_storage.filename)
    save_path = AppConfig.UPLOAD_DIR / safe_filename
    try:
        file_storage.stream.seek(0)
        file_storage.save(save_path)
    except OSError as exc:
        logger.error("Failed to save uploaded file: %s", exc)
        return {"success": False, "status": "error", "message": "Could not save the uploaded file."}, 500

    # Re-open the saved file for inference so we predict on exactly what was stored
    with open(save_path, "rb") as f:
        result = model_service.run_prediction_from_stream(f)

    status = result.get("status", "error")
    plant = result.get("plant")
    disease = result.get("disease")
    confidence = result.get("confidence")
    class_name = result.get("class_name")

    try:
        save_prediction_record(
            image_filename=safe_filename, plant=plant, disease=disease,
            confidence=confidence, status=status,
        )
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to save prediction history record: %s", exc)
        # Do not fail the whole request just because history logging failed

    response = {
        "success": status == "success",
        "status": status,
        "image_filename": safe_filename,
    }
    if status == "success":
        response.update({
            "plant": plant, "disease": disease, "confidence": confidence,
            "class_name": class_name,
        })
    else:
        response["message"] = result.get("message", "Prediction could not be completed.")
        if "confidence" in result:
            response["confidence"] = result["confidence"]

    http_status = 200 if status in ("success", "uncertain") else 500
    return response, http_status
