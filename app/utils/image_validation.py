"""
app/utils/image_validation.py

File-upload security and validation utilities:
- allowed extension check
- file size check
- secure, randomized filename generation
- lightweight image-integrity check (delegates to src/preprocessing)
"""

from __future__ import annotations

import uuid
from pathlib import Path

from werkzeug.utils import secure_filename

from app.config import AppConfig


class UploadValidationError(Exception):
    """Raised when an uploaded file fails validation."""


def has_allowed_extension(filename: str) -> bool:
    ext = Path(filename).suffix.lower()
    return ext in AppConfig.ALLOWED_EXTENSIONS


def generate_safe_filename(original_filename: str) -> str:
    """Combine a secure version of the original name with a random UUID
    prefix so stored filenames can never collide or be guessed."""
    ext = Path(original_filename).suffix.lower()
    safe_stem = secure_filename(Path(original_filename).stem) or "upload"
    return f"{uuid.uuid4().hex}_{safe_stem}{ext}"


def validate_upload(file_storage) -> None:
    """
    Raises UploadValidationError with a user-facing message if the upload
    is missing, has a disallowed extension, or exceeds the size limit.
    Flask's MAX_CONTENT_LENGTH already enforces the size limit at the
    request level; this is a defense-in-depth explicit check too.
    """
    if file_storage is None or file_storage.filename == "":
        raise UploadValidationError("No image was selected. Please choose a leaf image to upload.")

    if not has_allowed_extension(file_storage.filename):
        allowed = ", ".join(sorted(AppConfig.ALLOWED_EXTENSIONS))
        raise UploadValidationError(f"Unsupported file type. Allowed types: {allowed}")

    file_storage.stream.seek(0, 2)  # seek to end
    size = file_storage.stream.tell()
    file_storage.stream.seek(0)
    if size > AppConfig.MAX_CONTENT_LENGTH:
        max_mb = AppConfig.MAX_CONTENT_LENGTH // (1024 * 1024)
        raise UploadValidationError(f"Image is too large. Maximum allowed size is {max_mb} MB.")

    if size == 0:
        raise UploadValidationError("The uploaded file is empty.")
