"""
dataset_utils.py

Builds tf.data.Dataset objects from data/split/{train,val,test}/<Class>/.
Only the training dataset gets augmentation; validation/test stay
untouched apart from resize + rescale, exactly matching predict.py.
"""

from __future__ import annotations

import json
from pathlib import Path

from src.config_loader import load_config, resolve_path, get_image_size


def load_class_names() -> list[str]:
    cfg = load_config()
    class_names_path = resolve_path(cfg["paths"]["class_names_file"])
    if not class_names_path.exists():
        raise FileNotFoundError(
            f"{class_names_path} not found. Run `python -m src.split_dataset` first "
            "to generate the class list from your dataset."
        )
    with open(class_names_path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_datasets(batch_size: int | None = None, augment_train: bool = True):
    """
    Returns (train_ds, val_ds, test_ds, class_names).
    Raises FileNotFoundError with a clear message if data/split/ is missing
    (i.e. the user has not run data_cleaning + split_dataset yet).
    """
    import tensorflow as tf
    from src.preprocessing import build_augmentation_layer

    cfg = load_config()
    split_dir = resolve_path(cfg["paths"]["split_dataset"])
    img_h, img_w = get_image_size()
    batch_size = batch_size or cfg["training"]["batch_size"]
    seed = cfg["random_seed"]

    for subset in ("train", "val", "test"):
        if not (split_dir / subset).exists():
            raise FileNotFoundError(
                f"{split_dir / subset} not found. Run:\n"
                "  python -m src.data_cleaning\n"
                "  python -m src.split_dataset\n"
                "before training."
            )

    class_names = load_class_names()

    def _make(subset: str, shuffle: bool):
        ds = tf.keras.utils.image_dataset_from_directory(
            split_dir / subset,
            labels="inferred",
            label_mode="categorical",
            class_names=class_names,
            image_size=(img_h, img_w),
            batch_size=batch_size,
            shuffle=shuffle,
            seed=seed,
        )
        # Rescale [0,255] -> [0,1], identical to src/preprocessing.py
        ds = ds.map(lambda x, y: (x / 255.0, y), num_parallel_calls=tf.data.AUTOTUNE)
        return ds

    train_ds = _make("train", shuffle=True)
    val_ds = _make("val", shuffle=False)
    test_ds = _make("test", shuffle=False)

    if augment_train:
        aug = build_augmentation_layer()
        train_ds = train_ds.map(
            lambda x, y: (aug(x, training=True), y), num_parallel_calls=tf.data.AUTOTUNE
        )

    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.prefetch(tf.data.AUTOTUNE)
    test_ds = test_ds.prefetch(tf.data.AUTOTUNE)

    return train_ds, val_ds, test_ds, class_names
