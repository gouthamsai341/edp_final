"""
tests/test_prediction.py

Tests the prediction pipeline logic in src/predict.py WITHOUT requiring
a real trained model on disk, by monkeypatching load_model_and_classes
with a fake model that returns controlled probability vectors. This lets
us verify the confidence-threshold / uncertain-handling logic (Section 18
of the spec) deterministically.
"""

import io

import numpy as np
import pytest
from PIL import Image

import src.predict as predict_module


class _FakeModel:
    def __init__(self, probs):
        self._probs = np.array(probs, dtype=np.float32)

    def predict(self, tensor, verbose=0):
        return np.expand_dims(self._probs, axis=0)


def _fake_image_bytes():
    buf = io.BytesIO()
    Image.new("RGB", (32, 32), color=(50, 120, 60)).save(buf, format="JPEG")
    buf.seek(0)
    return buf


@pytest.fixture(autouse=True)
def reset_prediction_cache():
    predict_module._model = None
    predict_module._class_names = None
    yield
    predict_module._model = None
    predict_module._class_names = None


def test_high_confidence_prediction_returns_success(monkeypatch):
    class_names = ["Tomato___Early_blight", "Tomato___healthy"]
    fake_model = _FakeModel([0.95, 0.05])

    monkeypatch.setattr(
        predict_module, "load_model_and_classes",
        lambda model_path=None: (fake_model, class_names),
    )

    result = predict_module.predict_image(_fake_image_bytes())
    assert result["status"] == "success"
    assert result["plant"] == "Tomato"
    assert result["disease"] == "Early blight"
    assert result["confidence"] == pytest.approx(0.95)


def test_low_confidence_prediction_returns_uncertain(monkeypatch):
    class_names = ["Tomato___Early_blight", "Tomato___healthy"]
    fake_model = _FakeModel([0.52, 0.48])  # below default 0.60 threshold

    monkeypatch.setattr(
        predict_module, "load_model_and_classes",
        lambda model_path=None: (fake_model, class_names),
    )

    result = predict_module.predict_image(_fake_image_bytes())
    assert result["status"] == "uncertain"
    assert "message" in result


def test_invalid_image_returns_error(monkeypatch):
    class_names = ["Tomato___Early_blight", "Tomato___healthy"]
    fake_model = _FakeModel([0.9, 0.1])

    monkeypatch.setattr(
        predict_module, "load_model_and_classes",
        lambda model_path=None: (fake_model, class_names),
    )

    result = predict_module.predict_image(io.BytesIO(b"this is not an image"))
    assert result["status"] == "error"


def test_missing_model_returns_error(monkeypatch):
    def _raise(model_path=None):
        raise FileNotFoundError("no model")

    monkeypatch.setattr(predict_module, "load_model_and_classes", _raise)
    result = predict_module.predict_image(_fake_image_bytes())
    assert result["status"] == "error"
