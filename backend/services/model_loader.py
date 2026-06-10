"""
Universal Keras / TensorFlow model loader with fallback strategies.

Supports:
  - Keras 3 directory format (config.json + model.weights.h5)
  - Single .keras archive files
  - Standalone .h5 weight files (with architecture JSON)
  - TensorFlow SavedModel directories
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

import keras
import tensorflow as tf

logger = logging.getLogger(__name__)

_model_cache: tf.keras.Model | None = None
_model_metadata: dict[str, Any] = {}


def _resolve_model_dir(model_path: str | Path) -> Path:
    path = Path(model_path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Model path does not exist: {path}")
    return path


def _load_from_keras_dir(model_dir: Path) -> tf.keras.Model:
    config_path = model_dir / "config.json"
    weights_path = model_dir / "model.weights.h5"

    if not config_path.exists():
        raise FileNotFoundError(f"Missing config.json in {model_dir}")
    if not weights_path.exists():
        raise FileNotFoundError(f"Missing model.weights.h5 in {model_dir}")

    with open(config_path, encoding="utf-8") as f:
        config = json.load(f)

    model = keras.models.model_from_json(json.dumps(config))
    model.load_weights(str(weights_path))
    model.trainable = False
    logger.info("Loaded Keras directory model from %s", model_dir)
    return model


def _load_from_keras_file(model_file: Path) -> tf.keras.Model:
    model = keras.models.load_model(str(model_file), compile=False)
    model.trainable = False
    logger.info("Loaded .keras archive from %s", model_file)
    return model


def _load_from_h5(model_file: Path, config_path: Path | None = None) -> tf.keras.Model:
    if config_path and config_path.exists():
        with open(config_path, encoding="utf-8") as f:
            config = json.load(f)
        model = keras.models.model_from_json(json.dumps(config))
        model.load_weights(str(model_file))
    else:
        model = keras.models.load_model(str(model_file), compile=False)
    model.trainable = False
    logger.info("Loaded .h5 model from %s", model_file)
    return model


def _load_from_savedmodel(model_dir: Path) -> tf.keras.Model:
    model = tf.keras.models.load_model(str(model_dir), compile=False)
    model.trainable = False
    logger.info("Loaded SavedModel from %s", model_dir)
    return model


def discover_model_path(base_path: str | Path) -> Path:
    """Auto-discover the best model artifact under a path."""
    base = Path(base_path).resolve()

    if base.is_file():
        return base

    if (base / "config.json").exists() and (base / "model.weights.h5").exists():
        return base

    for name in ("saved_model.pb",):
        if (base / name).exists():
            return base

    for pattern in ("*.keras", "*.h5"):
        matches = sorted(base.glob(pattern))
        if matches:
            return matches[0]

    raise FileNotFoundError(f"No supported model artifact found at {base}")


def load_model(model_path: str | None = None, force_reload: bool = False) -> tf.keras.Model:
    """
    Load and cache the production deepfake detection model.

    Fallback order:
      1. Keras directory (config.json + weights)
      2. .keras archive
      3. .h5 with optional config.json
      4. SavedModel directory
    """
    global _model_cache, _model_metadata

    if _model_cache is not None and not force_reload:
        return _model_cache

    default_dir = Path(__file__).resolve().parent.parent / "models" / "model_finetuned_keras"
    resolved = discover_model_path(model_path or os.environ.get("MODEL_PATH", default_dir))

    try:
        if resolved.is_dir():
            if (resolved / "saved_model.pb").exists():
                _model_cache = _load_from_savedmodel(resolved)
            else:
                _model_cache = _load_from_keras_dir(resolved)
        elif resolved.suffix == ".keras":
            _model_cache = _load_from_keras_file(resolved)
        elif resolved.suffix == ".h5":
            config = resolved.parent / "config.json"
            _model_cache = _load_from_h5(resolved, config if config.exists() else None)
        else:
            raise ValueError(f"Unsupported model format: {resolved}")
    except Exception as primary_error:
        zip_path = Path(__file__).resolve().parent.parent / "model_finetuned.keras.zip"
        if zip_path.exists():
            import tempfile
            import zipfile

            logger.warning("Primary load failed (%s), trying zip archive fallback", primary_error)
            with tempfile.TemporaryDirectory() as tmp:
                with zipfile.ZipFile(zip_path) as zf:
                    zf.extractall(tmp)
                tmp_path = Path(tmp)
                if (tmp_path / "config.json").exists():
                    _model_cache = _load_from_keras_dir(tmp_path)
                else:
                    _model_cache = _load_from_keras_file(next(tmp_path.glob("*.keras")))
        else:
            raise primary_error

    _model_metadata = extract_model_metadata(_model_cache, resolved)
    return _model_cache


def extract_model_metadata(model: tf.keras.Model, source: Path) -> dict[str, Any]:
    """Infer input shape, output shape, backbone, and label mapping."""
    input_shape = tuple(model.input_shape[1:]) if model.input_shape else (224, 224, 3)
    output_shape = tuple(model.output_shape[1:]) if model.output_shape else (1,)

    backbone = "unknown"
    for layer in model.layers:
        name = layer.name.lower()
        if "mobilenet" in name or "efficientnet" in name or "resnet" in name:
            backbone = layer.name
            break

    return {
        "source": str(source),
        "input_shape": input_shape,
        "output_shape": output_shape,
        "backbone": backbone,
        "label_mapping": {0: "real", 1: "fake"},
        "preprocessing": "mobilenet_v2.preprocess_input",
        "num_layers": len(model.layers),
    }


def get_model_metadata() -> dict[str, Any]:
    if not _model_metadata:
        load_model()
    return dict(_model_metadata)


def get_backbone(model: tf.keras.Model) -> tf.keras.Model | None:
    """Return the nested backbone (e.g. MobileNetV2) if present."""
    if model.layers and hasattr(model.layers[0], "layers"):
        return model.layers[0]
    return None
