"""
train_cnn.py

Trains a custom CNN baseline (built from scratch, not pretrained) on the
split+augmented dataset. This satisfies the EDP Week 4 "set up CNN"
requirement and gives a real baseline to compare the transfer-learning
model against in evaluate.py.

Usage:
    python src/train_cnn.py
    python src/train_cnn.py --epochs 20 --batch-size 16
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from src.config_loader import load_config, resolve_path, get_image_size, get_random_seed

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def build_cnn_model(num_classes: int, input_shape: tuple[int, int, int]):
    import tensorflow as tf
    from tensorflow.keras import layers, models

    model = models.Sequential([
        layers.Input(shape=input_shape),

        layers.Conv2D(32, 3, padding="same"),
        layers.BatchNormalization(),
        layers.ReLU(),
        layers.MaxPooling2D(),

        layers.Conv2D(64, 3, padding="same"),
        layers.BatchNormalization(),
        layers.ReLU(),
        layers.MaxPooling2D(),

        layers.Conv2D(128, 3, padding="same"),
        layers.BatchNormalization(),
        layers.ReLU(),
        layers.MaxPooling2D(),

        layers.Conv2D(128, 3, padding="same"),
        layers.BatchNormalization(),
        layers.ReLU(),
        layers.GlobalAveragePooling2D(),

        layers.Dropout(0.3),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation="softmax"),
    ], name="cnn_baseline")

    return model


def main(epochs: int | None, batch_size: int | None):
    import tensorflow as tf
    from src.dataset_utils import build_datasets

    tf.random.set_seed(get_random_seed())
    cfg = load_config()
    epochs = epochs or cfg["training"]["epochs_cnn"]
    lr = cfg["training"]["learning_rate_cnn"]
    img_h, img_w = get_image_size()

    train_ds, val_ds, test_ds, class_names = build_datasets(batch_size=batch_size)
    num_classes = len(class_names)
    logger.info("Training CNN baseline on %d classes: %s", num_classes, class_names)

    model = build_cnn_model(num_classes, (img_h, img_w, 3))
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.summary(print_fn=logger.info)

    reports_dir = resolve_path(cfg["paths"]["reports_dir"])
    reports_dir.mkdir(parents=True, exist_ok=True)
    model_path = resolve_path(cfg["paths"]["cnn_model_path"])
    model_path.parent.mkdir(parents=True, exist_ok=True)

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=cfg["training"]["early_stopping_patience"],
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(model_path), monitor="val_accuracy", save_best_only=True,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=cfg["training"]["reduce_lr_factor"],
            patience=cfg["training"]["reduce_lr_patience"],
        ),
        tf.keras.callbacks.CSVLogger(str(reports_dir / "cnn_training_log.csv")),
    ]

    history = model.fit(train_ds, validation_data=val_ds, epochs=epochs, callbacks=callbacks)

    _plot_history(history.history, reports_dir, prefix="cnn")

    with open(reports_dir / "cnn_training_history.json", "w", encoding="utf-8") as f:
        json.dump(history.history, f, indent=2)

    logger.info("CNN baseline saved to %s", model_path)


def _plot_history(history: dict, reports_dir: Path, prefix: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if "accuracy" in history:
        plt.figure()
        plt.plot(history["accuracy"], label="train")
        plt.plot(history.get("val_accuracy", []), label="val")
        plt.title(f"{prefix} accuracy")
        plt.xlabel("epoch")
        plt.ylabel("accuracy")
        plt.legend()
        plt.savefig(reports_dir / f"{prefix}_training_accuracy.png", bbox_inches="tight")
        plt.close()

    if "loss" in history:
        plt.figure()
        plt.plot(history["loss"], label="train")
        plt.plot(history.get("val_loss", []), label="val")
        plt.title(f"{prefix} loss")
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.legend()
        plt.savefig(reports_dir / f"{prefix}_training_loss.png", bbox_inches="tight")
        plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the custom CNN baseline.")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    args = parser.parse_args()
    main(epochs=args.epochs, batch_size=args.batch_size)
