"""
Orchestrates the complete deepfake detection + XAI pipeline.
"""

from __future__ import annotations

import base64
import json
import logging
import os
import uuid
from datetime import datetime, timezone

import numpy as np

from config import MODEL_PATH, REPORTS_DIR, USE_BLIP, USE_GRADCAM
from services.blip_service import caption_region, rule_based_caption
from services.llm_service import generate_forensic_explanation
from services.model_loader import get_model_metadata, load_model
from services.social_safety import assess_social_risks
from utils.forensic_analysis import run_forensic_analysis
from utils.gradcam import (
    annotate_image_with_regions,
    compute_gradcam_plusplus,
    extract_suspicious_regions,
    overlay_heatmap,
)
from utils.preprocessing import preprocess_with_noise_robustness
from utils.trust_score import compute_trust_score

logger = logging.getLogger(__name__)


def get_model():
    return load_model(MODEL_PATH)


def analyze_image(img_rgb: np.ndarray) -> dict:
    """Full pipeline analysis returning a forensic report dict."""
    report_id = str(uuid.uuid4())[:8]
    model = get_model()
    metadata = get_model_metadata()

    preprocessed = preprocess_with_noise_robustness(img_rgb)
    raw_pred = float(model.predict(preprocessed, verbose=0)[0][0])

    prediction = "fake" if raw_pred >= 0.5 else "real"
    confidence = raw_pred if prediction == "fake" else (1.0 - raw_pred)

    heatmap_b64 = None
    annotated_b64 = None
    regions: list[dict] = []
    mean_heatmap_intensity = 0.0
    gradcam_layer = None

    if USE_GRADCAM:
        try:
            class_idx = 0 if prediction == "fake" else 0
            heatmap = compute_gradcam_plusplus(model, preprocessed, class_idx=class_idx)
            _, heatmap_png = overlay_heatmap(img_rgb, heatmap)
            heatmap_b64 = base64.b64encode(heatmap_png).decode()

            regions = extract_suspicious_regions(img_rgb, heatmap)
            if regions:
                mean_heatmap_intensity = float(
                    np.mean([r["intensity"] for r in regions])
                )
            annotated_b64 = base64.b64encode(
                annotate_image_with_regions(img_rgb, regions)
            ).decode()
            gradcam_layer = "out_relu"
        except Exception as exc:
            logger.warning("Grad-CAM failed, continuing without heatmap: %s", exc)

    forensic = run_forensic_analysis(img_rgb)
    fft_b64 = base64.b64encode(forensic["fft_image_bytes"]).decode()

    region_captions: list[str] = []
    suspicious_regions_out: list[dict] = []

    for i, reg in enumerate(regions):
        if USE_BLIP:
            try:
                caption = caption_region(reg["crop_rgb"])
            except Exception:
                caption = rule_based_caption(reg)
        else:
            caption = rule_based_caption(reg)

        region_captions.append(caption)
        suspicious_regions_out.append({
            "index": i + 1,
            "bbox": list(reg["bbox"]),
            "intensity": reg["intensity"],
            "area": reg.get("area", 0),
            "caption": caption,
        })

    ts_result = compute_trust_score(
        confidence=confidence,
        prediction=prediction,
        mean_heatmap_intensity=mean_heatmap_intensity,
        num_suspicious_regions=len(regions),
        fft_anomaly_score=forensic["fft_analysis"]["fft_anomaly_score"],
        captions=region_captions,
    )

    social_risks = assess_social_risks(
        prediction=prediction,
        confidence=confidence,
        trust_score=ts_result["trust_score"],
        suspicion_score=forensic["suspicion_score"],
        num_regions=len(regions),
    )

    forensic_explanation = generate_forensic_explanation(
        prediction=prediction,
        confidence=confidence,
        trust_score=ts_result["trust_score"],
        risk_level=ts_result["risk_level"],
        region_captions=region_captions,
        fft_anomaly_score=forensic["fft_analysis"]["fft_anomaly_score"],
        high_freq_energy=forensic["fft_analysis"]["high_freq_energy"],
        num_regions=len(regions),
        mean_heatmap_intensity=mean_heatmap_intensity,
        manipulation_indicators=forensic["manipulation_indicators"],
        authenticity_score=forensic["authenticity_score"],
    )

    misuse_warning = None
    if social_risks["warnings"]:
        misuse_warning = " ".join(social_risks["warnings"][:2])
    elif prediction == "fake":
        misuse_warning = (
            "Potential misuse in cyberbullying, impersonation, or misinformation."
        )

    report = {
        "report_id": report_id,
        "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "prediction": prediction.capitalize(),
        "confidence": round(confidence * 100, 2),
        "raw_score": round(raw_pred, 4),
        "trust_score": ts_result["trust_score"],
        "risk_level": ts_result["risk_level"],
        "score_components": ts_result["components"],
        "heatmap_image": heatmap_b64,
        "annotated_image": annotated_b64,
        "fft_image": fft_b64,
        "fft_analysis": forensic["fft_analysis"],
        "texture_analysis": forensic["texture_analysis"],
        "noise_analysis": forensic["noise_analysis"],
        "compression_analysis": forensic["compression_analysis"],
        "forensic_scores": {
            "authenticity_score": forensic["authenticity_score"],
            "suspicion_score": forensic["suspicion_score"],
            "forensic_confidence": forensic["forensic_confidence"],
            "manipulation_indicators": forensic["manipulation_indicators"],
        },
        "social_safety": social_risks,
        "suspicious_regions": suspicious_regions_out,
        "region_captions": region_captions,
        "forensic_explanation": forensic_explanation,
        "misuse_warning": misuse_warning,
        "model_info": {
            "architecture": f"{metadata.get('backbone', 'MobileNetV2')} + GlobalAveragePooling + Dense",
            "input_size": "224×224",
            "gradcam_layer": gradcam_layer,
            "label_mapping": metadata.get("label_mapping", {0: "real", 1: "fake"}),
        },
    }

    report_path = os.path.join(REPORTS_DIR, f"{report_id}.json")
    disk_report = {
        k: v
        for k, v in report.items()
        if k not in ("heatmap_image", "annotated_image", "fft_image")
    }
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(disk_report, f, indent=2)

    return report
