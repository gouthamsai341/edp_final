"""
app/config.py

Loads settings from the project-level config.yaml (single source of
truth shared with the ML pipeline in src/) and exposes a Flask
configuration object. Secrets (SECRET_KEY) come from an environment
variable, never hard-coded.
"""

from __future__ import annotations

import os
from pathlib import Path

from src.config_loader import load_config, resolve_path

_cfg = load_config()

BASE_DIR = Path(__file__).resolve().parent.parent


class AppConfig:
    SECRET_KEY = os.environ.get(_cfg["flask"]["secret_key_env_var"], "dev-only-insecure-key-change-me")
    DEBUG = _cfg["flask"]["debug"]

    UPLOAD_DIR = resolve_path(_cfg["paths"]["upload_dir"])
    ALLOWED_EXTENSIONS = set(_cfg["upload"]["allowed_extensions"])
    MAX_CONTENT_LENGTH = int(_cfg["upload"]["max_size_mb"]) * 1024 * 1024

    DB_PATH = resolve_path(_cfg["paths"]["database_path"])
    SQLALCHEMY_DATABASE_URI = f"sqlite:///{DB_PATH}"
    SQLALCHEMY_TRACK_MODIFICATIONS = False

    DISEASE_INFO_PATH = resolve_path(_cfg["paths"]["disease_info"])
    CONFIDENCE_THRESHOLD = _cfg["inference"]["confidence_threshold"]

    HOST = _cfg["flask"]["host"]
    PORT = _cfg["flask"]["port"]


def ensure_runtime_dirs() -> None:
    AppConfig.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    AppConfig.DB_PATH.parent.mkdir(parents=True, exist_ok=True)
