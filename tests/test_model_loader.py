"""Tests for universal model loader."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

from services.model_loader import extract_model_metadata, load_model  # noqa: E402


@pytest.fixture(scope="module")
def model():
    return load_model(force_reload=True)


def test_model_loads(model):
    assert model is not None
    assert len(model.layers) >= 3


def test_model_metadata(model):
    model_path = os.path.join(
        os.path.dirname(__file__), "..", "backend", "models", "model_finetuned_keras"
    )
    from pathlib import Path
    meta = extract_model_metadata(model, Path(model_path))
    assert meta["input_shape"] == (224, 224, 3)
    assert meta["label_mapping"][0] == "real"
    assert meta["label_mapping"][1] == "fake"


def test_model_inference(model):
    from keras import applications

    x = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8).astype(np.float32)
    x = applications.mobilenet_v2.preprocess_input(x)
    x = np.expand_dims(x, 0)
    pred = model.predict(x, verbose=0)
    assert pred.shape == (1, 1)
    assert 0.0 <= float(pred[0][0]) <= 1.0
