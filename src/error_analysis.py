"""
error_analysis.py

Finds every misclassified image in the test set and writes a CSV report
(actual class, predicted class, confidence, image path), plus optionally
copies a sample of misclassified images into reports/misclassified_samples/
for quick visual inspection. This helps identify which diseases the
model confuses with each other (e.g. visually similar blights).

Usage:
    python src/error_analysis.py
    python src/error_analysis.py --model models/cnn_baseline.keras --save-samples 20
"""

from __future__ import annotations

import argparse
import logging
import shutil
from pathlib import Path

import numpy as np

from src.config_loader import load_config, resolve_path, get_image_size

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main(model_path_arg: str | None, save_samples: int):
    import pandas as pd
    import tensorflow as tf
    from src.dataset_utils import load_class_names
    from src.preprocessing import load_and_preprocess_image, batch_to_tensor, InvalidImageError

    cfg = load_config()
    reports_dir = resolve_path(cfg["paths"]["reports_dir"])
    reports_dir.mkdir(parents=True, exist_ok=True)

    model_path = resolve_path(model_path_arg) if model_path_arg else resolve_path(cfg["paths"]["best_model_path"])
    if not model_path.exists():
        logger.error("No trained model found at %s. Train and evaluate a model first.", model_path)
        return

    class_names = load_class_names()
    model = tf.keras.models.load_model(model_path)

    test_dir = resolve_path(cfg["paths"]["split_dataset"]) / "test"
    if not test_dir.exists():
        logger.error("Test split not found at %s. Run src/split_dataset.py first.", test_dir)
        return

    img_size = get_image_size()
    rows = []

    for class_idx, class_name in enumerate(class_names):
        class_dir = test_dir / class_name
        if not class_dir.exists():
            continue
        for file_path in sorted(class_dir.iterdir()):
            if not file_path.is_file():
                continue
            try:
                arr = load_and_preprocess_image(file_path, img_size)
            except InvalidImageError:
                continue
            probs = model.predict(batch_to_tensor(arr), verbose=0)[0]
            pred_idx = int(np.argmax(probs))
            if pred_idx != class_idx:
                rows.append({
                    "actual_class": class_name,
                    "predicted_class": class_names[pred_idx],
                    "confidence": float(probs[pred_idx]),
                    "image_path": str(file_path),
                })

    df = pd.DataFrame(rows)
    out_csv = reports_dir / "misclassified_report.csv"
    df.to_csv(out_csv, index=False)
    logger.info("Found %d misclassified test images. Report: %s", len(df), out_csv)

    if not df.empty:
        confusion_pairs = (
            df.groupby(["actual_class", "predicted_class"])
            .size()
            .reset_index(name="count")
            .sort_values("count", ascending=False)
        )
        confusion_pairs.to_csv(reports_dir / "top_confusion_pairs.csv", index=False)
        logger.info("Top confused class pairs written to reports/top_confusion_pairs.csv")

    if save_samples > 0 and not df.empty:
        sample_dir = reports_dir / "misclassified_samples"
        if sample_dir.exists():
            shutil.rmtree(sample_dir)
        sample_dir.mkdir(parents=True)
        for i, row in df.head(save_samples).iterrows():
            src = Path(row["image_path"])
            dst_name = f"{i}_actual-{row['actual_class']}_pred-{row['predicted_class']}{src.suffix}"
            shutil.copy2(src, sample_dir / dst_name)
        logger.info("Saved %d misclassified sample images to %s", min(save_samples, len(df)), sample_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze misclassified test images.")
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--save-samples", type=int, default=10)
    args = parser.parse_args()
    main(model_path_arg=args.model, save_samples=args.save_samples)
