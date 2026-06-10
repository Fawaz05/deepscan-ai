"""Grad-CAM tests."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

from keras import applications  # noqa: E402
from services.model_loader import load_model  # noqa: E402
from utils.gradcam import compute_gradcam_plusplus  # noqa: E402


@pytest.fixture(scope="module")
def model():
    return load_model(force_reload=True)


def test_gradcam_output_shape(model):
    x = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8).astype(np.float32)
    x = applications.mobilenet_v2.preprocess_input(x)
    x = np.expand_dims(x, 0)
    heatmap = compute_gradcam_plusplus(model, x)
    assert heatmap.ndim == 2
    assert heatmap.min() >= 0.0
    assert heatmap.max() <= 1.0 + 1e-5
