"""
app/services/model_service.py

Thin wrapper around src/predict.py that ensures the model is loaded ONCE
when the Flask app starts (see app/app.py's create_app), not on every
request, per the "Performance" requirement in the project spec.
"""

from __future__ import annotations

import logging

from src.predict import load_model_and_classes, predict_image

logger = logging.getLogger(__name__)

_model_ready = False


def warm_up_model() -> bool:
    """Attempt to load the model and class names at startup. Returns True
    on success, False if no trained model exists yet (the app still runs -
    predictions will return a friendly 'model unavailable' error until a
    model is trained and placed in models/)."""
    global _model_ready
    try:
        load_model_and_classes()
        _model_ready = True
        logger.info("Model loaded and ready for inference.")
    except FileNotFoundError as exc:
        _model_ready = False
        logger.warning(
            "No trained model available yet (%s). The /api/predict endpoint "
            "will return a friendly error until a model is trained. See README.",
            exc,
        )
    except Exception as exc:  # noqa: BLE001 - e.g. TensorFlow not installed yet
        _model_ready = False
        logger.warning(
            "Model could not be loaded at startup (%s). The /api/predict endpoint "
            "will return a friendly error until this is resolved.",
            exc,
        )
    return _model_ready


def is_model_ready() -> bool:
    return _model_ready


def run_prediction_from_stream(file_obj) -> dict:
    """Delegates to the shared src/predict.py pipeline. file_obj is any
    open, seekable, binary file-like object (e.g. an open() handle or a
    Werkzeug FileStorage.stream)."""
    if not _model_ready:
        return {
            "status": "error",
            "message": (
                "The prediction model is not available yet. Train a model "
                "(see README 'Training the Model') and restart the app."
            ),
        }
    return predict_image(file_obj)
