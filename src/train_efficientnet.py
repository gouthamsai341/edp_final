"""
train_efficientnet.py

Transfer learning with EfficientNetB0 (ImageNet weights), in two stages:

  Stage 1 (head training): freeze the backbone, train a new classifier
  head on top.

  Stage 2 (fine-tuning): unfreeze the top N backbone layers and continue
  training with a much lower learning rate.

This is the model saved as models/best_plant_disease_model.keras and is
the one src/predict.py and the Flask app load, UNLESS evaluate.py shows
the CNN baseline actually performs better on the held-out test set (see
evaluate.py's model-selection note in the README).

Usage:
    python src/train_efficientnet.py
    python src/train_efficientnet.py --epochs-head 10 --epochs-finetune 10
"""

from __future__ import annotations

import argparse
import json
import logging

import numpy as np

from src.config_loader import load_config, resolve_path, get_image_size, get_random_seed
from src.train_cnn import _plot_history  # reuse the plotting helper

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def build_efficientnet_model(num_classes: int, input_shape: tuple[int, int, int]):
    import tensorflow as tf
    from tensorflow.keras.applications import EfficientNetB0
    from tensorflow.keras import layers, models

    backbone = EfficientNetB0(
        include_top=False, weights="imagenet", input_shape=input_shape, pooling="avg"
    )
    backbone.trainable = False  # Stage 1: frozen backbone

    inputs = layers.Input(shape=input_shape)
    # EfficientNet expects pixels in [0, 255]; our pipeline provides [0, 1],
    # so rescale back up before EfficientNet's own preprocess_input.
    x = layers.Rescaling(255.0)(inputs)
    x = tf.keras.applications.efficientnet.preprocess_input(x)
    x = backbone(x, training=False)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inputs, outputs, name="efficientnet_transfer")
    return model, backbone


def compute_class_weights(train_ds, num_classes: int) -> dict[int, float]:
    """Compute class weights from the training set to counter class imbalance."""
    counts = np.zeros(num_classes, dtype=np.int64)
    for _, labels in train_ds.unbatch():
        counts[np.argmax(labels.numpy())] += 1
    counts = np.maximum(counts, 1)
    total = counts.sum()
    weights = total / (num_classes * counts)
    return {i: float(w) for i, w in enumerate(weights)}


def main(epochs_head: int | None, epochs_finetune: int | None, batch_size: int | None):
    import tensorflow as tf
    from src.dataset_utils import build_datasets

    tf.random.set_seed(get_random_seed())
    cfg = load_config()
    epochs_head = epochs_head or cfg["training"]["epochs_head"]
    epochs_finetune = epochs_finetune or cfg["training"]["epochs_finetune"]
    img_h, img_w = get_image_size()

    train_ds, val_ds, test_ds, class_names = build_datasets(batch_size=batch_size)
    num_classes = len(class_names)
    logger.info("Training EfficientNetB0 transfer model on %d classes", num_classes)

    model, backbone = build_efficientnet_model(num_classes, (img_h, img_w, 3))

    reports_dir = resolve_path(cfg["paths"]["reports_dir"])
    reports_dir.mkdir(parents=True, exist_ok=True)
    model_path = resolve_path(cfg["paths"]["best_model_path"])
    model_path.parent.mkdir(parents=True, exist_ok=True)

    class_weights = compute_class_weights(train_ds, num_classes)
    logger.info("Class weights: %s", class_weights)

    # ---------- Stage 1: train the head, backbone frozen ----------
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=cfg["training"]["learning_rate_head"]),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    callbacks_stage1 = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=cfg["training"]["early_stopping_patience"],
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=cfg["training"]["reduce_lr_factor"],
            patience=cfg["training"]["reduce_lr_patience"],
        ),
        tf.keras.callbacks.CSVLogger(str(reports_dir / "efficientnet_stage1_log.csv")),
    ]
    history1 = model.fit(
        train_ds, validation_data=val_ds, epochs=epochs_head,
        class_weight=class_weights, callbacks=callbacks_stage1,
    )

    # ---------- Stage 2: unfreeze top layers, fine-tune ----------
    backbone.trainable = True
    n_unfreeze = cfg["training"]["fine_tune_unfreeze_layers"]
    for layer in backbone.layers[:-n_unfreeze]:
        layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=cfg["training"]["learning_rate_finetune"]),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    callbacks_stage2 = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=cfg["training"]["early_stopping_patience"],
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(model_path), monitor="val_accuracy", save_best_only=True,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=cfg["training"]["reduce_lr_factor"],
            patience=cfg["training"]["reduce_lr_patience"],
        ),
        tf.keras.callbacks.CSVLogger(str(reports_dir / "efficientnet_stage2_log.csv")),
    ]
    history2 = model.fit(
        train_ds, validation_data=val_ds, epochs=epochs_finetune,
        class_weight=class_weights, callbacks=callbacks_stage2,
    )

    # If ModelCheckpoint never triggered (e.g. val_accuracy never improved
    # past its initial -inf baseline in a short run), save explicitly.
    if not model_path.exists():
        model.save(model_path)

    combined_history = {
        k: history1.history.get(k, []) + history2.history.get(k, [])
        for k in set(history1.history) | set(history2.history)
    }
    _plot_history(combined_history, reports_dir, prefix="efficientnet")
    with open(reports_dir / "efficientnet_training_history.json", "w", encoding="utf-8") as f:
        json.dump(combined_history, f, indent=2)

    logger.info("EfficientNetB0 model saved to %s", model_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train EfficientNetB0 transfer-learning model.")
    parser.add_argument("--epochs-head", type=int, default=None)
    parser.add_argument("--epochs-finetune", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    args = parser.parse_args()
    main(epochs_head=args.epochs_head, epochs_finetune=args.epochs_finetune, batch_size=args.batch_size)
