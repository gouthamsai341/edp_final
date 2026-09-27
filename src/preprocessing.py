"""
preprocessing.py

Single source of truth for image preprocessing so that training and the
Flask inference service can never drift apart. Both src/train_*.py and
src/predict.py import from this module.

Pipeline: validate -> RGB convert -> resize -> normalize -> tensor
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, UnidentifiedImageError

from src.config_loader import get_image_size


class InvalidImageError(Exception):
    """Raised when an uploaded/loaded file cannot be treated as a valid image."""


def load_and_preprocess_image(path_or_file, target_size: tuple[int, int] | None = None) -> np.ndarray:
    """
    Load an image from a path or file-like object, convert to RGB, resize
    to the model's expected input size, and scale pixel values to [0, 1].

    Returns a numpy array of shape (H, W, 3), dtype float32.
    """
    target_size = target_size or get_image_size()
    try:
        img = Image.open(path_or_file)
        img.verify()
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise InvalidImageError(f"File is not a valid image: {exc}") from exc

    # Re-open after verify() (verify() leaves the file unusable for further ops)
    if isinstance(path_or_file, (str, Path)):
        img = Image.open(path_or_file)
    else:
        path_or_file.seek(0)
        img = Image.open(path_or_file)

    img = img.convert("RGB")
    img = img.resize((target_size[1], target_size[0]))  # PIL uses (W, H)

    array = np.asarray(img, dtype=np.float32) / 255.0
    return array


def batch_to_tensor(image_array: np.ndarray) -> np.ndarray:
    """Add the batch dimension expected by model.predict()."""
    return np.expand_dims(image_array, axis=0)


def build_augmentation_layer():
    """
    Returns a tf.keras Sequential of augmentation layers to be applied
    ONLY to the training dataset (never validation/test). Imports
    TensorFlow lazily so that non-training code (e.g. the Flask app) does
    not pay the TF import cost.
    """
    import tensorflow as tf
    from src.config_loader import load_config

    cfg = load_config()["augmentation"]
    layers = []
    if cfg.get("horizontal_flip"):
        layers.append(tf.keras.layers.RandomFlip("horizontal"))
    rotation = cfg.get("rotation_range_deg", 0) / 360.0
    if rotation > 0:
        layers.append(tf.keras.layers.RandomRotation(rotation))
    zoom = cfg.get("zoom_range", 0)
    if zoom > 0:
        layers.append(tf.keras.layers.RandomZoom(zoom))
    translation = cfg.get("translation_range", 0)
    if translation > 0:
        layers.append(tf.keras.layers.RandomTranslation(translation, translation))
    contrast = cfg.get("contrast_range", 0)
    if contrast > 0:
        layers.append(tf.keras.layers.RandomContrast(contrast))

    return tf.keras.Sequential(layers, name="train_time_augmentation")
