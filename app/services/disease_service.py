"""
app/services/disease_service.py

Loads and serves the disease information library from
data/disease_info.json. Cached in memory after first load since the
file is static reference content.
"""

from __future__ import annotations

import json
import logging

from app.config import AppConfig

logger = logging.getLogger(__name__)

_disease_info_cache: dict | None = None


def _load() -> dict:
    global _disease_info_cache
    if _disease_info_cache is None:
        if not AppConfig.DISEASE_INFO_PATH.exists():
            logger.warning("disease_info.json not found at %s", AppConfig.DISEASE_INFO_PATH)
            _disease_info_cache = {}
        else:
            with open(AppConfig.DISEASE_INFO_PATH, "r", encoding="utf-8") as f:
                _disease_info_cache = json.load(f)
    return _disease_info_cache


def get_all_diseases() -> dict:
    return _load()


def get_disease_by_id(class_name: str) -> dict | None:
    return _load().get(class_name)


def find_disease_key_for(plant: str | None, disease: str | None) -> str | None:
    """Best-effort lookup of a disease_info.json key matching a predicted
    (plant, disease) pair whose class-name formatting may differ slightly
    (e.g. underscores vs spaces)."""
    if not plant or not disease:
        return None
    normalized_target = f"{plant}_{disease}".lower().replace(" ", "_")
    for key in _load():
        if key.lower().replace(" ", "_") == normalized_target:
            return key
        # Loose match: both plant and disease substrings present
        key_norm = key.lower()
        if plant.lower().replace(" ", "_") in key_norm and disease.lower().split()[0] in key_norm:
            return key
    return None
