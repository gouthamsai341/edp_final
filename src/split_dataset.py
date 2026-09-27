"""
split_dataset.py

Performs a STRATIFIED train/validation/test split on the *cleaned*
dataset (data/cleaned/<ClassName>/*), copying files into:

    data/split/train/<ClassName>/...
    data/split/val/<ClassName>/...
    data/split/test/<ClassName>/...

This step happens BEFORE any augmentation, which is applied only to the
train/ directory at training time (see src/preprocessing.py). This
ordering is the key defence against data leakage: validation and test
images are never touched except by deterministic preprocessing.

A fixed random seed (config.yaml -> random_seed) makes the split
reproducible.

Run:
    python -m src.split_dataset
"""

from __future__ import annotations

import json
import logging
import random
import shutil
from pathlib import Path
from typing import Dict, List

from src.config_loader import load_config, resolve_path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def split_class_files(files: List[Path], train_ratio: float, val_ratio: float, seed: int):
    rng = random.Random(seed)
    shuffled = files[:]
    rng.shuffle(shuffled)

    n = len(shuffled)
    n_train = int(round(n * train_ratio))
    n_val = int(round(n * val_ratio))

    train_files = shuffled[:n_train]
    val_files = shuffled[n_train:n_train + n_val]
    test_files = shuffled[n_train + n_val:]
    return train_files, val_files, test_files


def run_split() -> dict:
    cfg = load_config()
    cleaned_dir = resolve_path(cfg["paths"]["cleaned_dataset"])
    split_dir = resolve_path(cfg["paths"]["split_dataset"])
    seed = cfg["random_seed"]
    train_ratio = cfg["split"]["train_ratio"]
    val_ratio = cfg["split"]["val_ratio"]
    test_ratio = cfg["split"]["test_ratio"]

    assert abs((train_ratio + val_ratio + test_ratio) - 1.0) < 1e-6, \
        "split ratios in config.yaml must sum to 1.0"

    if not cleaned_dir.exists() or not any(cleaned_dir.iterdir()):
        logger.warning(
            "No cleaned dataset found at %s. Run `python -m src.data_cleaning` first.",
            cleaned_dir,
        )
        return {"message": "No cleaned dataset found."}

    classes = sorted([p.name for p in cleaned_dir.iterdir() if p.is_dir()])
    summary: Dict[str, Dict[str, int]] = {}

    for class_name in classes:
        class_folder = cleaned_dir / class_name
        files = sorted([p for p in class_folder.iterdir() if p.is_file()])
        if not files:
            continue

        train_files, val_files, test_files = split_class_files(
            files, train_ratio, val_ratio, seed
        )

        for split_name, split_files in (
            ("train", train_files),
            ("val", val_files),
            ("test", test_files),
        ):
            out_dir = split_dir / split_name / class_name
            out_dir.mkdir(parents=True, exist_ok=True)
            for f in split_files:
                shutil.copy2(f, out_dir / f.name)

        summary[class_name] = {
            "total": len(files),
            "train": len(train_files),
            "val": len(val_files),
            "test": len(test_files),
        }
        logger.info(
            "%s -> train=%d val=%d test=%d",
            class_name, len(train_files), len(val_files), len(test_files),
        )

    reports_dir = resolve_path(cfg["paths"]["reports_dir"])
    reports_dir.mkdir(parents=True, exist_ok=True)
    with open(reports_dir / "split_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # Persist the class list (order matters - this becomes the model's label order)
    models_dir = resolve_path(cfg["paths"]["models_dir"])
    models_dir.mkdir(parents=True, exist_ok=True)
    class_names_path = resolve_path(cfg["paths"]["class_names_file"])
    with open(class_names_path, "w", encoding="utf-8") as f:
        json.dump(classes, f, indent=2)

    logger.info("Split complete. Classes written to %s", class_names_path)
    return summary


if __name__ == "__main__":
    run_split()
