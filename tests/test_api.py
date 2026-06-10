"""API integration tests."""

import os
import sys

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

from main import app  # noqa: E402

client = TestClient(app)


def test_root():
    resp = client.get("/")
    assert resp.status_code == 200
    assert resp.json()["service"] == "DeepScan AI"


def test_health():
    resp = client.get("/api/v1/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"


def test_reports_list():
    resp = client.get("/api/v1/reports")
    assert resp.status_code == 200
    assert "reports" in resp.json()


def _make_test_image_bytes() -> bytes:
    arr = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    img = Image.fromarray(arr)
    from io import BytesIO
    buf = BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


@pytest.mark.slow
def test_upload_image():
    data = _make_test_image_bytes()
    resp = client.post(
        "/api/v1/upload-image",
        files={"file": ("test.png", data, "image/png")},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert "report_id" in body
    assert body["prediction"] in ("Real", "Fake")
    assert 0 <= body["confidence"] <= 100
