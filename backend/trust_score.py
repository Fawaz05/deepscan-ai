"""
utils/trust_score.py
--------------------
Trust Score Engine (0–100).
Combines model confidence, heatmap intensity, region count,
FFT anomaly score, and caption severity.
"""

import re


SEVERITY_KEYWORDS = [
    "manipulation", "artifact", "inconsistent", "unnatural", "blending",
    "distortion", "anomaly", "irregular", "fake", "synthetic", "warping",
    "ghosting", "compression", "seam", "boundary", "mismatch",
]


def caption_severity_score(captions: list[str]) -> float:
    """
    Count how many severity keywords appear across all captions.
    Returns a 0–1 score.
    """
    if not captions:
        return 0.0
    text = " ".join(captions).lower()
    hits = sum(1 for kw in SEVERITY_KEYWORDS if kw in text)
    return min(1.0, hits / max(1, len(captions) * 3))


def compute_trust_score(
    confidence: float,          # model output probability (0–1) for "fake"
    prediction: str,            # "fake" or "real"
    mean_heatmap_intensity: float,   # mean of top-region heatmap intensities
    num_suspicious_regions: int,
    fft_anomaly_score: float,   # 0–100
    captions: list[str],
) -> dict:
    """
    Calculate an overall Trust Score and Risk Level.

    Trust Score:  0 = definitely fake / untrusted
                  100 = definitely real / trusted

    When prediction == "fake", trust score should be LOW.
    When prediction == "real", trust score should be HIGH.
    """
    cap_severity = caption_severity_score(captions)

    if prediction == "fake":
        # Contributions pull score DOWN (toward 0)
        model_component    = confidence                         # 0–1, high → less trust
        heatmap_component  = min(1.0, mean_heatmap_intensity)
        region_component   = min(1.0, num_suspicious_regions / 5.0)
        fft_component      = fft_anomaly_score / 100.0
        caption_component  = cap_severity

        # Weighted average of distrust signals
        distrust = (
            0.40 * model_component
            + 0.20 * heatmap_component
            + 0.15 * region_component
            + 0.15 * fft_component
            + 0.10 * caption_component
        )
        trust_score = round((1.0 - distrust) * 100, 1)
    else:
        # prediction == "real"
        # Model says real — start from high trust, penalise anomalies
        real_confidence = 1.0 - confidence  # how confident model is in "real"
        penalty = (
            0.25 * min(1.0, mean_heatmap_intensity)
            + 0.15 * min(1.0, num_suspicious_regions / 5.0)
            + 0.15 * (fft_anomaly_score / 100.0)
            + 0.05 * cap_severity
        )
        trust_score = round((real_confidence * 0.55 + (1.0 - penalty) * 0.45) * 100, 1)

    trust_score = max(0.0, min(100.0, trust_score))

    # Risk level thresholds
    if trust_score < 30:
        risk_level = "High"
    elif trust_score < 60:
        risk_level = "Medium"
    else:
        risk_level = "Low"

    return {
        "trust_score":  trust_score,
        "risk_level":   risk_level,
        "components": {
            "model_confidence":   round(confidence * 100, 1),
            "heatmap_intensity":  round(mean_heatmap_intensity * 100, 1),
            "regions_detected":   num_suspicious_regions,
            "fft_anomaly":        fft_anomaly_score,
            "caption_severity":   round(cap_severity * 100, 1),
        },
    }
