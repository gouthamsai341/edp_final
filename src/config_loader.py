"""
config_loader.py

Loads config.yaml once and exposes it as a plain dict, plus a couple of
convenience helpers. Every other script (data cleaning, training,
evaluation, prediction, and the Flask app) imports from here instead of
re-reading YAML or hard-coding paths/hyperparameters.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict

import yaml

# Project root = parent of this file's parent (src/ -> project root)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "config.yaml"

_config_cache: Dict[str, Any] | None = None


def load_config() -> Dict[str, Any]:
    """Load config.yaml once and cache it for subsequent calls."""
    global _config_cache
    if _config_cache is None:
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            _config_cache = yaml.safe_load(f)
    return _config_cache


def resolve_path(relative_path: str) -> Path:
    """Resolve a path from config.yaml relative to the project root."""
    return PROJECT_ROOT / relative_path


def get_image_size() -> tuple[int, int]:
    cfg = load_config()
    h, w = cfg["image"]["size"]
    return int(h), int(w)


def get_random_seed() -> int:
    return int(load_config()["random_seed"])
