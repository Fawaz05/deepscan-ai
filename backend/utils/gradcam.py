"""
utils/gradcam.py
----------------
Grad-CAM++ implementation for MobileNetV2 inside a Sequential wrapper.
Also provides region extraction (contour-based) for anomalous areas.
"""

import numpy as np
import cv2
import tensorflow as tf
import io
from PIL import Image


# Last convolutional layer in MobileNetV2
GRADCAM_LAYER_NAME = "out_relu"   # final ReLU in MobileNetV2 backbone


# ─────────────────────────────────────────────
# Grad-CAM++ core
# ─────────────────────────────────────────────

def _find_target_layer(model):
    """Locate the last conv activation layer, including nested backbones."""
    search_models = [model]
    if model.layers and hasattr(model.layers[0], "layers"):
        search_models.insert(0, model.layers[0])

    for submodel in search_models:
        try:
            return submodel.get_layer(GRADCAM_LAYER_NAME)
        except ValueError:
            pass

        for layer in reversed(submodel.layers):
            if isinstance(layer, (tf.keras.layers.Conv2D, tf.keras.layers.ReLU)):
                if "out" in layer.name.lower() or isinstance(layer, tf.keras.layers.Conv2D):
                    return layer

    raise RuntimeError("No suitable convolutional layer found for Grad-CAM")


def _ensure_model_built(model, input_shape: tuple[int, int, int] = (224, 224, 3)) -> None:
    """Keras 3 Sequential models need a forward pass before graph introspection."""
    from keras import applications

    dummy = np.zeros((1, *input_shape), dtype=np.float32)
    dummy = applications.mobilenet_v2.preprocess_input(dummy[0])
    dummy = np.expand_dims(dummy, 0)
    model.predict(dummy, verbose=0)


def get_gradcam_model(model):
    """
    Build a sub-model that exposes both the last conv feature maps
    and the final prediction. Handles nested backbone (MobileNetV2) wrappers.
    """
    _ensure_model_built(model)
    conv_layer = _find_target_layer(model)

    if model.layers and hasattr(model.layers[0], "layers"):
        backbone = model.layers[0]
        inp = backbone.inputs[0]
        predictions = model(inp)
        gradcam_model = tf.keras.Model(
            inputs=inp,
            outputs=[conv_layer.output, predictions],
        )
    else:
        inp = model.inputs[0]
        gradcam_model = tf.keras.Model(
            inputs=inp,
            outputs=[conv_layer.output, model(inp)],
        )
    return gradcam_model


def _compute_gradcam_standard(
    gradcam_model,
    inp: tf.Tensor,
    class_idx: int,
) -> np.ndarray:
    """Standard Grad-CAM (first-order gradients)."""
    with tf.GradientTape() as tape:
        conv_outputs, predictions = gradcam_model(inp)
        tape.watch(conv_outputs)
        score = predictions[:, class_idx]

    grads = tape.gradient(score, conv_outputs)
    if grads is None:
        raise RuntimeError("Grad-CAM gradient computation returned None")

    grads = grads.numpy()[0]
    conv_outputs = conv_outputs.numpy()[0]
    weights = np.mean(grads, axis=(0, 1))
    heatmap = np.maximum(np.sum(conv_outputs * weights, axis=-1), 0)

    if heatmap.max() > 0:
        heatmap /= heatmap.max()
    return heatmap.astype(np.float32)


def _compute_gradcam_plusplus_core(
    gradcam_model,
    inp: tf.Tensor,
    class_idx: int,
) -> np.ndarray:
    """Grad-CAM++ with higher-order gradients."""
    with tf.GradientTape() as tape2:
        with tf.GradientTape() as tape1:
            with tf.GradientTape() as tape0:
                conv_outputs, predictions = gradcam_model(inp)
                tape0.watch(conv_outputs)
                tape1.watch(conv_outputs)
                tape2.watch(conv_outputs)
                score = predictions[:, class_idx]

            grads = tape0.gradient(score, conv_outputs)
        second_grads = tape1.gradient(grads, conv_outputs)

    third_grads = tape2.gradient(second_grads, conv_outputs)

    if grads is None or second_grads is None or third_grads is None:
        raise RuntimeError("Grad-CAM++ gradient computation returned None")

    grads = grads.numpy()[0]
    second_grads = second_grads.numpy()[0]
    third_grads = third_grads.numpy()[0]
    conv_outputs = conv_outputs.numpy()[0]

    numerator = second_grads ** 2
    denominator = 2 * second_grads ** 2 + conv_outputs * third_grads + 1e-7
    alpha = numerator / denominator
    weights = np.sum(alpha * np.maximum(grads, 0), axis=(0, 1))
    weights = np.maximum(weights, 0)

    heatmap = np.zeros(conv_outputs.shape[:2], dtype=np.float32)
    for i, w in enumerate(weights):
        heatmap += w * conv_outputs[:, :, i]

    heatmap = np.maximum(heatmap, 0)
    if heatmap.max() > 0:
        heatmap /= heatmap.max()
    return heatmap


def _compute_cam_from_weights(model, preprocessed_input: np.ndarray, class_idx: int = 0) -> np.ndarray:
    """
    Weight-based Class Activation Map — reliable fallback when gradient-based
    Grad-CAM is unavailable (e.g. Keras 3 frozen nested models on TF 2.21).
    """
    backbone = model.layers[0]
    conv_layer = _find_target_layer(model)
    feat_model = tf.keras.Model(
        inputs=backbone.inputs,
        outputs=conv_layer.output,
    )
    feats = feat_model.predict(preprocessed_input, verbose=0)[0]

    # Propagate classifier weights back to conv channels
    weights = None
    dense_layers = [l for l in model.layers if isinstance(l, tf.keras.layers.Dense)]
    if len(dense_layers) >= 2:
        w1 = dense_layers[0].get_weights()[0]
        w2 = dense_layers[-1].get_weights()[0]
        weights = w1 @ w2
    elif len(dense_layers) == 1:
        weights = dense_layers[0].get_weights()[0]

    if weights is None:
        weights = np.mean(feats, axis=(0, 1), keepdims=True).reshape(-1, 1)

    channel_weights = weights[:, class_idx] if weights.ndim == 2 else weights
    heatmap = np.maximum(np.sum(feats * channel_weights, axis=-1), 0)

    if heatmap.max() > 0:
        heatmap /= heatmap.max()
    return heatmap.astype(np.float32)


def compute_gradcam_plusplus(
    model,
    preprocessed_input: np.ndarray,
    class_idx: int = 0,
) -> np.ndarray:
    """
    Compute Grad-CAM++ heatmap with fallbacks:
      1. Grad-CAM++ (higher-order gradients)
      2. Standard Grad-CAM
      3. Weight-based CAM (always works with GAP/Dense heads)

    Returns:
        heatmap – float32 array in [0, 1], shape (H, W)
    """
    gradcam_model = get_gradcam_model(model)
    inp = tf.cast(preprocessed_input, tf.float32)

    try:
        return _compute_gradcam_plusplus_core(gradcam_model, inp, class_idx)
    except (RuntimeError, TypeError):
        try:
            return _compute_gradcam_standard(gradcam_model, inp, class_idx)
        except (RuntimeError, TypeError):
            return _compute_cam_from_weights(model, preprocessed_input, class_idx)


# ─────────────────────────────────────────────
# Heatmap overlay
# ─────────────────────────────────────────────

def overlay_heatmap(
    original_rgb: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.45,
    colormap: int = cv2.COLORMAP_JET,
) -> tuple[np.ndarray, bytes]:
    """
    Resize heatmap to image size, apply colormap, blend with original.

    Returns:
        overlaid_rgb  – numpy uint8 (H, W, 3)
        png_bytes     – PNG bytes of the overlay
    """
    h, w = original_rgb.shape[:2]
    heatmap_resized = cv2.resize(heatmap, (w, h), interpolation=cv2.INTER_CUBIC)
    heatmap_uint8   = np.uint8(255 * heatmap_resized)
    heatmap_colored = cv2.applyColorMap(heatmap_uint8, colormap)
    heatmap_rgb     = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

    overlaid = cv2.addWeighted(
        original_rgb.astype(np.uint8), 1 - alpha,
        heatmap_rgb,                   alpha,
        0,
    )

    _, buf = cv2.imencode(".png", cv2.cvtColor(overlaid, cv2.COLOR_RGB2BGR))
    return overlaid, buf.tobytes()


# ─────────────────────────────────────────────
# Region extraction
# ─────────────────────────────────────────────

def extract_suspicious_regions(
    original_rgb: np.ndarray,
    heatmap: np.ndarray,
    threshold: float = 0.5,
    top_k: int = 5,
    min_area: int = 400,
) -> list[dict]:
    """
    Threshold the heatmap, find contours, extract the top-k largest
    suspicious bounding-box crops from the original image.

    Returns list of dicts with keys:
        bbox       – (x, y, w, h)
        crop_rgb   – numpy array of the cropped region
        crop_bytes – PNG bytes
        intensity  – mean heatmap intensity in the region
    """
    h, w = original_rgb.shape[:2]
    heatmap_resized = cv2.resize(heatmap, (w, h))

    # Threshold
    binary = (heatmap_resized >= threshold).astype(np.uint8) * 255

    # Morphological cleanup
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN,  kernel)

    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    regions = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue
        x, y, bw, bh = cv2.boundingRect(cnt)
        # Add small padding
        pad = 10
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(w, x + bw + pad)
        y2 = min(h, y + bh + pad)

        crop = original_rgb[y1:y2, x1:x2]
        mean_intensity = float(heatmap_resized[y1:y2, x1:x2].mean())

        _, buf = cv2.imencode(".png", cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))

        regions.append({
            "bbox":       (x1, y1, x2 - x1, y2 - y1),
            "crop_rgb":   crop,
            "crop_bytes": buf.tobytes(),
            "intensity":  round(mean_intensity, 3),
            "area":       int(area),
        })

    # Sort by intensity descending, return top-k
    regions.sort(key=lambda r: r["intensity"], reverse=True)
    return regions[:top_k]


def annotate_image_with_regions(
    original_rgb: np.ndarray,
    regions: list[dict],
) -> bytes:
    """
    Draw bounding boxes around suspicious regions on the original image.
    Returns PNG bytes.
    """
    annotated = original_rgb.copy()
    for i, reg in enumerate(regions):
        x, y, bw, bh = reg["bbox"]
        cv2.rectangle(annotated, (x, y), (x + bw, y + bh), (255, 50, 50), 3)
        label = f"Region {i+1} ({reg['intensity']:.2f})"
        cv2.putText(
            annotated, label, (x, max(0, y - 8)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 50, 50), 2,
        )
    _, buf = cv2.imencode(".png", cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))
    return buf.tobytes()