"""
tests/conftest.py

Shared pytest fixtures: a Flask test client backed by a temporary SQLite
database, and a tiny in-memory valid JPEG for upload tests.
"""

import io
import os
import sys
from pathlib import Path

import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

os.environ.setdefault("PLANTCARE_SECRET_KEY", "test-secret-key")


@pytest.fixture()
def app(tmp_path, monkeypatch):
    from app.config import AppConfig

    # Redirect DB + uploads to a temp directory so tests never touch real data
    monkeypatch.setattr(AppConfig, "DB_PATH", tmp_path / "test.db")
    monkeypatch.setattr(AppConfig, "SQLALCHEMY_DATABASE_URI", f"sqlite:///{tmp_path / 'test.db'}")
    monkeypatch.setattr(AppConfig, "UPLOAD_DIR", tmp_path / "uploads")

    from app.app import create_app
    flask_app = create_app()
    flask_app.config.update(TESTING=True)
    yield flask_app


@pytest.fixture()
def client(app):
    return app.test_client()


@pytest.fixture()
def valid_image_bytes():
    """A small, genuinely valid in-memory JPEG image for upload tests."""
    buf = io.BytesIO()
    img = Image.new("RGB", (64, 64), color=(80, 160, 90))
    img.save(buf, format="JPEG")
    buf.seek(0)
    return buf
