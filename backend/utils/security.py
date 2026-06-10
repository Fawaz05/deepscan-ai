"""Security utilities for input and file validation."""

from __future__ import annotations

from io import BytesIO

from fastapi import HTTPException, UploadFile
from PIL import Image

ALLOWED_CONTENT_TYPES = {
    "image/jpeg",
    "image/png",
    "image/webp",
    "image/bmp",
    "image/gif",
}

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"}


def validate_image_upload(file: UploadFile) -> None:
    if not file.filename:
        raise HTTPException(400, detail="Filename is required.")

    ext = "." + file.filename.rsplit(".", 1)[-1].lower() if "." in file.filename else ""
    if ext and ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(400, detail=f"Unsupported file extension: {ext}")

    if file.content_type and not file.content_type.startswith("image/"):
        if file.content_type not in ALLOWED_CONTENT_TYPES:
            raise HTTPException(400, detail="File must be an image (JPEG, PNG, WEBP, etc.).")


def validate_image_bytes(data: bytes) -> None:
    if len(data) < 32:
        raise HTTPException(400, detail="File is too small to be a valid image.")

    try:
        with Image.open(BytesIO(data)) as img:
            img.verify()
    except Exception as exc:
        raise HTTPException(422, detail="Could not verify image format from file contents.") from exc
