"""
services/inference_engine.py
-----------------------------
Orchestrates the complete deepfake detection + XAI pipeline:
  1. Load model (once, cached)
  2. Preprocess image
  3. Model inference
  4. Grad-CAM++ heatmap
  5. Region extraction
  6. FFT analysis
  7. BLIP region captioning
  8. LLM forensic explanation
  9. Trust score
 10. Assemble forensic report
"""

import os
import uuid
import base64
import json
import numpy as np
from datetime import datetime

import tensorflow as tf

from utils.preprocessing import (
    preprocess_with_noise_robustness,
    compute_fft_features,
    compute_texture_features,
)
from utils.gradcam import (
    compute_gradcam_plusplus,
    overlay_heatmap,
    extract_suspicious_regions,
    annotate_image_with_regions,
)
from utils.trust_score import compute_trust_score
from services.blip_service import caption_region, rule_based_caption
from services.llm_service import generate_forensic_explanation

import keras


# ─────────────────────────────────────────────
# Singleton model
# ─────────────────────────────────────────────

_model = None
MODEL_DIR = os.environ.get(
    "MODEL_DIR",
    os.path.join(os.path.dirname(__file__), "..", "models", "model_finetuned_keras"),
)


def get_model():
    global _model
    if _model is None:
        print(f"[Engine] Loading model from {MODEL_DIR} ...")
        config_path  = os.path.join(MODEL_DIR, "config.json")
        weights_path = os.path.join(MODEL_DIR, "model.weights.h5")
        with open(config_path) as f:
            import json as _json
            config = _json.load(f)
        _model = keras.models.model_from_json(json.dumps(config))
        _model.load_weights(weights_path)
        _model.trainable = False
        print("[Engine] Model loaded and ready.")
    return _model


# ─────────────────────────────────────────────
# Settings
# ─────────────────────────────────────────────

USE_BLIP = os.environ.get("USE_BLIP", "false").lower() == "true"
REPORTS_DIR = os.environ.get("REPORTS_DIR", "reports")
os.makedirs(REPORTS_DIR, exist_ok=True)


# ─────────────────────────────────────────────
# Main analysis function
# ─────────────────────────────────────────────

def analyze_image(img_rgb: np.ndarray) -> dict:
    """
    Full pipeline analysis.

    Args:
        img_rgb: HxWx3 uint8 RGB numpy array (original, any size)

    Returns:
        Forensic report dict
    """
    report_id = str(uuid.uuid4())[:8]
    model = get_model()

    # 1. Preprocess
    preprocessed = preprocess_with_noise_robustness(img_rgb)  # (1, 224, 224, 3)

    # 2. Inference
    raw_pred = float(model.predict(preprocessed, verbose=0)[0][0])

    # Determine prediction
    # Output neuron: 0 = real, 1 = fake  (adjust if your labels differ)
    prediction  = "fake" if raw_pred >= 0.5 else "real"
    confidence  = raw_pred if prediction == "fake" else (1.0 - raw_pred)

    # 3. Grad-CAM++
    heatmap = compute_gradcam_plusplus(model, preprocessed, class_idx=0)

    overlaid_rgb, heatmap_overlay_bytes = overlay_heatmap(img_rgb, heatmap, alpha=0.45)
    heatmap_b64 = base64.b64encode(heatmap_overlay_bytes).decode()

    # 4. Region extraction
    regions = extract_suspicious_regions(
        img_rgb, heatmap, threshold=0.45, top_k=5, min_area=400
    )
    annotated_bytes = annotate_image_with_regions(img_rgb, regions)
    annotated_b64   = base64.b64encode(annotated_bytes).decode()

    mean_heatmap_intensity = (
        float(np.mean([r["intensity"] for r in regions])) if regions else 0.0
    )

    # 5. FFT analysis
    fft_results  = compute_fft_features(img_rgb)
    fft_b64      = base64.b64encode(fft_results["fft_image_bytes"]).decode()
    texture_info = compute_texture_features(img_rgb)

    # 6. Region captioning
    region_captions = []
    for reg in regions[:3]:   # caption top 3 most suspicious regions
        if USE_BLIP:
            try:
                cap = caption_region(reg["crop_rgb"])
            except Exception:
                cap = rule_based_caption(reg)
        else:
            cap = rule_based_caption(reg)
        region_captions.append(cap)

    # 7. Trust score
    ts_result = compute_trust_score(
        confidence=confidence,
        prediction=prediction,
        mean_heatmap_intensity=mean_heatmap_intensity,
        num_suspicious_regions=len(regions),
        fft_anomaly_score=fft_results["fft_anomaly_score"],
        captions=region_captions,
    )

    # 8. LLM forensic explanation
    forensic_explanation = generate_forensic_explanation(
        prediction=prediction,
        confidence=confidence,
        trust_score=ts_result["trust_score"],
        risk_level=ts_result["risk_level"],
        region_captions=region_captions,
        fft_anomaly_score=fft_results["fft_anomaly_score"],
        high_freq_energy=fft_results["high_freq_energy"],
        num_regions=len(regions),
        mean_heatmap_intensity=mean_heatmap_intensity,
    )

    # 9. Cyberbullying / misinformation warning
    misuse_warning = (
        "⚠ Potential misuse in cyberbullying, impersonation, or misinformation."
        if prediction == "fake"
        else None
    )

    # 10. Assemble report
    report = {
        "report_id":    report_id,
        "timestamp":    datetime.utcnow().isoformat() + "Z",
        "prediction":   prediction.capitalize(),
        "confidence":   round(confidence * 100, 2),
        "raw_score":    round(raw_pred, 4),
        "trust_score":  ts_result["trust_score"],
        "risk_level":   ts_result["risk_level"],
        "score_components": ts_result["components"],
        "heatmap_image":    heatmap_b64,
        "annotated_image":  annotated_b64,
        "fft_image":        fft_b64,
        "fft_analysis": {
            "high_freq_energy": fft_results["high_freq_energy"],
            "fft_anomaly_score": fft_results["fft_anomaly_score"],
        },
        "texture_analysis": texture_info,
        "suspicious_regions": [
            {
                "bbox":      list(r["bbox"]),
                "intensity": r["intensity"],
                "area":      r["area"],
                "caption":   region_captions[i] if i < len(region_captions) else "",
            }
            for i, r in enumerate(regions)
        ],
        "region_captions":      region_captions,
        "forensic_explanation": forensic_explanation,
        "misuse_warning":       misuse_warning,
        "model_info": {
            "architecture": "MobileNetV2 + GlobalAveragePooling + Dense",
            "input_size":   "224×224",
            "gradcam_layer": "out_relu",
        },
    }

    # Cache report to disk
    report_path = os.path.join(REPORTS_DIR, f"{report_id}.json")
    with open(report_path, "w") as f:
        # Don't serialise large b64 strings to disk report — keep only metadata
        disk_report = {k: v for k, v in report.items()
                       if k not in ("heatmap_image", "annotated_image", "fft_image")}
        json.dump(disk_report, f, indent=2)

    return report
