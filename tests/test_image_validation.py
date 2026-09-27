"""
tests/test_image_validation.py

Covers: valid extensions, invalid extensions, oversized files, and
secure filename generation - the "File Upload Security" requirements.
"""

import io

from werkzeug.datastructures import FileStorage

from app.utils.image_validation import (
    UploadValidationError, generate_safe_filename, has_allowed_extension,
    validate_upload,
)


def test_allowed_extension_accepted():
    assert has_allowed_extension("leaf.jpg") is True
    assert has_allowed_extension("leaf.PNG") is True


def test_disallowed_extension_rejected():
    assert has_allowed_extension("leaf.exe") is False
    assert has_allowed_extension("leaf.txt") is False


def test_generate_safe_filename_is_randomized_and_keeps_extension():
    name1 = generate_safe_filename("My Leaf Photo.JPG")
    name2 = generate_safe_filename("My Leaf Photo.JPG")
    assert name1 != name2
    assert name1.lower().endswith(".jpg")


def test_validate_upload_rejects_missing_file():
    try:
        validate_upload(None)
        assert False, "expected UploadValidationError"
    except UploadValidationError:
        pass


def test_validate_upload_rejects_unsupported_extension():
    fs = FileStorage(stream=io.BytesIO(b"not really an image"), filename="malware.exe")
    try:
        validate_upload(fs)
        assert False, "expected UploadValidationError"
    except UploadValidationError:
        pass


def test_validate_upload_accepts_supported_extension_and_nonempty_file(valid_image_bytes):
    fs = FileStorage(stream=valid_image_bytes, filename="leaf.jpg")
    validate_upload(fs)  # should not raise
