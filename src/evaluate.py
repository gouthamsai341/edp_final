"""
evaluate.py

Loads a trained model and the held-out test set, runs real inference,
and writes out:

  reports/classification_report.csv
  reports/per_class_metrics.csv
  reports/per_class_accuracy.csv
  reports/confusion_matrix.png
  reports/test_metrics.json

If data/external_test/<Class>/ contains images, it is evaluated
SEPARATELY and written to reports/external_test_metrics.json, clearly
distinguished from PlantVillage-style internal test performance (see
README "Real-World Testing" section) - it is never mixed into training
or into the internal test numbers.

Nothing here is fabricated: every number comes from model.predict() on
real images. If no trained model or split dataset exists yet, this
script exits with a clear explanation instead of inventing results.

Usage:
    python src/evaluate.py --model models/best_plant_disease_model.keras
    python src/evaluate.py --model models/cnn_baseline.keras
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np

from src.config_loader import load_config, resolve_path, get_image_size

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def _predict_over_dataset(model, class_names: list[str], data_dir: Path):
    """Run model.predict over every image in data_dir/<ClassName>/*, returning
    (y_true, y_pred, confidences, filepaths). Uses the exact same preprocessing
    as training/inference (src/preprocessing.py) - no separate code path."""
    from src.preprocessing import load_and_preprocess_image, batch_to_tensor, InvalidImageError

    img_size = get_image_size()
    y_true, y_pred, confidences, filepaths = [], [], [], []

    for class_idx, class_name in enumerate(class_names):
        class_dir = data_dir / class_name
        if not class_dir.exists():
            continue
        for file_path in sorted(class_dir.iterdir()):
            if not file_path.is_file():
                continue
            try:
                arr = load_and_preprocess_image(file_path, img_size)
            except InvalidImageError:
                logger.warning("Skipping unreadable file during evaluation: %s", file_path)
                continue
            probs = model.predict(batch_to_tensor(arr), verbose=0)[0]
            pred_idx = int(np.argmax(probs))
            y_true.append(class_idx)
            y_pred.append(pred_idx)
            confidences.append(float(probs[pred_idx]))
            filepaths.append(str(file_path))

    return np.array(y_true), np.array(y_pred), np.array(confidences), filepaths


def _write_reports(y_true, y_pred, class_names, reports_dir: Path, prefix: str):
    import pandas as pd
    from sklearn.metrics import (
        classification_report, confusion_matrix, accuracy_score,
        precision_recall_fscore_support,
    )
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if len(y_true) == 0:
        logger.warning("No samples available for '%s' evaluation - skipping report generation.", prefix)
        return None

    overall_accuracy = accuracy_score(y_true, y_pred)

    report_dict = classification_report(
        y_true, y_pred, labels=list(range(len(class_names))),
        target_names=class_names, output_dict=True, zero_division=0,
    )
    pd.DataFrame(report_dict).transpose().to_csv(reports_dir / f"{prefix}_classification_report.csv")

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(len(class_names))), zero_division=0
    )
    per_class_df = pd.DataFrame({
        "class": class_names,
        "support": support,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
    })
    per_class_df.to_csv(reports_dir / f"{prefix}_per_class_metrics.csv", index=False)

    # Per-class accuracy (correct / total samples of that true class)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    with np.errstate(divide="ignore", invalid="ignore"):
        per_class_acc = np.divide(
            cm.diagonal(), cm.sum(axis=1),
            out=np.zeros(len(class_names)), where=cm.sum(axis=1) != 0,
        )
    pd.DataFrame({
        "class": class_names,
        "test_samples": cm.sum(axis=1),
        "correct": cm.diagonal(),
        "accuracy": per_class_acc,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
    }).to_csv(reports_dir / f"{prefix}_per_class_accuracy.csv", index=False)

    # Confusion matrix plot, sized to stay readable with many classes
    n = len(class_names)
    fig_size = max(8, n * 0.4)
    plt.figure(figsize=(fig_size, fig_size))
    plt.imshow(cm, interpolation="nearest", cmap="Greens")
    plt.title(f"{prefix} confusion matrix")
    plt.colorbar()
    tick_marks = np.arange(n)
    plt.xticks(tick_marks, class_names, rotation=90, fontsize=max(4, 10 - n // 10))
    plt.yticks(tick_marks, class_names, fontsize=max(4, 10 - n // 10))
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig(reports_dir / f"{prefix}_confusion_matrix.png", dpi=150)
    plt.close()

    metrics_summary = {
        "accuracy": float(overall_accuracy),
        "macro_avg": {
            "precision": float(report_dict["macro avg"]["precision"]),
            "recall": float(report_dict["macro avg"]["recall"]),
            "f1_score": float(report_dict["macro avg"]["f1-score"]),
        },
        "weighted_avg": {
            "precision": float(report_dict["weighted avg"]["precision"]),
            "recall": float(report_dict["weighted avg"]["recall"]),
            "f1_score": float(report_dict["weighted avg"]["f1-score"]),
        },
        "num_samples": int(len(y_true)),
        "num_classes": len(class_names),
    }
    with open(reports_dir / f"{prefix}_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics_summary, f, indent=2)

    logger.info("%s overall accuracy: %.4f (n=%d)", prefix, overall_accuracy, len(y_true))
    return metrics_summary


def main(model_path_arg: str | None):
    import tensorflow as tf
    from src.dataset_utils import load_class_names

    cfg = load_config()
    reports_dir = resolve_path(cfg["paths"]["reports_dir"])
    reports_dir.mkdir(parents=True, exist_ok=True)

    model_path = resolve_path(model_path_arg) if model_path_arg else resolve_path(cfg["paths"]["best_model_path"])
    if not model_path.exists():
        logger.error(
            "No trained model found at %s. Train a model first:\n"
            "  python src/train_cnn.py\n"
            "  python src/train_efficientnet.py",
            model_path,
        )
        return

    class_names = load_class_names()
    model = tf.keras.models.load_model(model_path)

    split_dir = resolve_path(cfg["paths"]["split_dataset"])
    test_dir = split_dir / "test"
    if not test_dir.exists():
        logger.error("Test split not found at %s. Run src/split_dataset.py first.", test_dir)
        return

    logger.info("Evaluating on internal (PlantVillage-style) test set...")
    y_true, y_pred, confidences, filepaths = _predict_over_dataset(model, class_names, test_dir)
    _write_reports(y_true, y_pred, class_names, reports_dir, prefix="test")

    # Real-world / external test set, evaluated and reported SEPARATELY
    external_dir = resolve_path(cfg["paths"]["external_test"])
    if external_dir.exists() and any(external_dir.iterdir()):
        logger.info("Evaluating on external/real-world test set (kept separate from internal metrics)...")
        y_true_ext, y_pred_ext, conf_ext, files_ext = _predict_over_dataset(model, class_names, external_dir)
        _write_reports(y_true_ext, y_pred_ext, class_names, reports_dir, prefix="external_test")
    else:
        logger.info(
            "No external test images found at %s. "
            "Add real-world photos there (organized by class folder) to measure "
            "real-world generalization separately from PlantVillage performance.",
            external_dir,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a trained model on the test set.")
    parser.add_argument("--model", type=str, default=None, help="Path to .keras model (default: best_plant_disease_model.keras)")
    args = parser.parse_args()
    main(model_path_arg=args.model)
