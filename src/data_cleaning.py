"""
data_cleaning.py

Scans data/raw/<ClassName>/*.jpg (etc.), validates every image, converts
to RGB, detects corrupt/unreadable/too-small files and near-duplicates
(via perceptual hashing), and produces a reproducible cleaning report.

Design choices (see README "Data Leakage Prevention" section):
- Nothing is silently deleted. Invalid/duplicate files are MOVED to
  data/quarantine/<reason>/<ClassName>/<file> so nothing is lost.
- Valid, deduplicated images are copied (RGB-converted) into
  data/cleaned/<ClassName>/, preserving the original filename.
- Class list is discovered dynamically from the folder names under
  data/raw/ - nothing is hard-coded.

Run:
    python -m src.data_cleaning
"""

from __future__ import annotations

import json
import logging
import shutil
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

import imagehash
from PIL import Image, UnidentifiedImageError

from src.config_loader import load_config, resolve_path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class ClassCleaningStats:
    class_name: str
    total_found: int = 0
    valid: int = 0
    corrupt: int = 0
    too_small: int = 0
    unsupported_ext: int = 0
    duplicates: int = 0
    dimensions_sample: List[List[int]] = field(default_factory=list)


def discover_classes(raw_dir: Path) -> List[str]:
    """Dynamically detect class folder names under data/raw/."""
    if not raw_dir.exists():
        return []
    return sorted([p.name for p in raw_dir.iterdir() if p.is_dir()])


def is_supported_extension(path: Path, allowed_extensions: List[str]) -> bool:
    return path.suffix.lower() in allowed_extensions


def compute_phash(path: Path, hash_size: int) -> imagehash.ImageHash | None:
    try:
        with Image.open(path) as img:
            return imagehash.phash(img, hash_size=hash_size)
    except Exception:
        return None


def clean_class_folder(
    class_name: str,
    raw_dir: Path,
    cleaned_dir: Path,
    quarantine_dir: Path,
    cfg: dict,
) -> ClassCleaningStats:
    stats = ClassCleaningStats(class_name=class_name)
    allowed_ext = cfg["cleaning"]["allowed_extensions"]
    min_w = cfg["cleaning"]["min_width"]
    min_h = cfg["cleaning"]["min_height"]
    hash_size = cfg["cleaning"]["duplicate_hash_size"]
    dup_threshold = cfg["cleaning"]["duplicate_hamming_threshold"]

    src_folder = raw_dir / class_name
    dst_folder = cleaned_dir / class_name
    dst_folder.mkdir(parents=True, exist_ok=True)

    seen_hashes: Dict[imagehash.ImageHash, Path] = {}

    for file_path in sorted(src_folder.iterdir()):
        if not file_path.is_file():
            continue
        stats.total_found += 1

        if not is_supported_extension(file_path, allowed_ext):
            stats.unsupported_ext += 1
            _quarantine(file_path, quarantine_dir / "unsupported_ext" / class_name)
            continue

        try:
            with Image.open(file_path) as img:
                img.verify()  # cheap structural check
            with Image.open(file_path) as img:
                width, height = img.size
                if width < min_w or height < min_h:
                    stats.too_small += 1
                    _quarantine(file_path, quarantine_dir / "too_small" / class_name)
                    continue
                stats.dimensions_sample.append([width, height])
                rgb_img = img.convert("RGB")
        except (UnidentifiedImageError, OSError, ValueError):
            stats.corrupt += 1
            _quarantine(file_path, quarantine_dir / "corrupt" / class_name)
            continue

        # Perceptual-hash duplicate detection within this class folder
        phash = compute_phash(file_path, hash_size)
        is_duplicate = False
        if phash is not None:
            for existing_hash in seen_hashes:
                if phash - existing_hash <= dup_threshold:
                    is_duplicate = True
                    break
        if is_duplicate:
            stats.duplicates += 1
            _quarantine(file_path, quarantine_dir / "duplicate" / class_name)
            continue
        if phash is not None:
            seen_hashes[phash] = file_path

        # Save the cleaned, RGB-converted image
        out_path = dst_folder / file_path.name
        rgb_img.save(out_path)
        stats.valid += 1

    return stats


def _quarantine(file_path: Path, quarantine_subdir: Path) -> None:
    quarantine_subdir.mkdir(parents=True, exist_ok=True)
    try:
        shutil.copy2(file_path, quarantine_subdir / file_path.name)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not quarantine %s: %s", file_path, exc)


def run_cleaning() -> dict:
    cfg = load_config()
    raw_dir = resolve_path(cfg["paths"]["raw_dataset"])
    cleaned_dir = resolve_path(cfg["paths"]["cleaned_dataset"])
    quarantine_dir = resolve_path(cfg["paths"]["quarantine_dir"])
    reports_dir = resolve_path(cfg["paths"]["reports_dir"])
    reports_dir.mkdir(parents=True, exist_ok=True)

    classes = discover_classes(raw_dir)
    if not classes:
        logger.warning(
            "No class folders found under %s. "
            "Populate data/raw/<ClassName>/ with images before cleaning. "
            "See README for dataset download instructions.",
            raw_dir,
        )
        return {"classes": [], "message": "No dataset found."}

    all_stats: Dict[str, ClassCleaningStats] = {}
    for class_name in classes:
        logger.info("Cleaning class: %s", class_name)
        all_stats[class_name] = clean_class_folder(
            class_name, raw_dir, cleaned_dir, quarantine_dir, cfg
        )

    report = {
        "classes_found": classes,
        "num_classes": len(classes),
        "per_class": {
            name: {
                "total_found": s.total_found,
                "valid": s.valid,
                "corrupt": s.corrupt,
                "too_small": s.too_small,
                "unsupported_ext": s.unsupported_ext,
                "duplicates": s.duplicates,
            }
            for name, s in all_stats.items()
        },
        "totals": {
            "total_found": sum(s.total_found for s in all_stats.values()),
            "valid": sum(s.valid for s in all_stats.values()),
            "corrupt": sum(s.corrupt for s in all_stats.values()),
            "too_small": sum(s.too_small for s in all_stats.values()),
            "unsupported_ext": sum(s.unsupported_ext for s in all_stats.values()),
            "duplicates": sum(s.duplicates for s in all_stats.values()),
        },
    }

    report_path = reports_dir / "dataset_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    logger.info("Cleaning complete. Report written to %s", report_path)
    return report


if __name__ == "__main__":
    run_cleaning()
