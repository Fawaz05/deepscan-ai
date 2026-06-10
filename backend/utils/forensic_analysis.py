"""
Professional forensic analysis engine.

Provides FFT, noise, texture, and compression artifact analysis
with authenticity and suspicion scoring.
"""

from __future__ import annotations

import numpy as np
import cv2

from utils.preprocessing import compute_fft_features, compute_texture_features, IMG_SIZE


def analyze_noise_patterns(img_rgb: np.ndarray) -> dict:
    """Estimate noise level and uniformity across image quadrants."""
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    h, w = gray.shape
    quadrants = {
        "top_left": gray[: h // 2, : w // 2],
        "top_right": gray[: h // 2, w // 2 :],
        "bottom_left": gray[h // 2 :, : w // 2],
        "bottom_right": gray[h // 2 :, w // 2 :],
    }

    noise_levels = {}
    for name, region in quadrants.items():
        lap = cv2.Laplacian(region, cv2.CV_64F)
        noise_levels[name] = float(lap.var())

    values = list(noise_levels.values())
    mean_noise = float(np.mean(values))
    std_noise = float(np.std(values))
    uniformity = 1.0 - min(1.0, std_noise / (mean_noise + 1e-6))

    suspicion = min(100.0, max(0.0, (1.0 - uniformity) * 60 + (mean_noise > 500) * 20))

    return {
        "quadrant_noise": {k: round(v, 2) for k, v in noise_levels.items()},
        "mean_noise": round(mean_noise, 2),
        "noise_uniformity": round(uniformity, 3),
        "noise_suspicion_score": round(suspicion, 1),
    }


def analyze_compression_artifacts(img_rgb: np.ndarray) -> dict:
    """
    Detect JPEG blocking artifacts via 8x8 DCT block boundary analysis.
    """
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    gray = cv2.resize(gray, IMG_SIZE).astype(np.float32)

    # Block boundary differences (align slice lengths)
    h_left = gray[:, 7:-1:8].astype(np.float32)
    h_right = gray[:, 8::8].astype(np.float32)
    min_h = min(h_left.shape[1], h_right.shape[1])
    h_diff = np.abs(h_left[:, :min_h] - h_right[:, :min_h])

    v_top = gray[7:-1:8, :].astype(np.float32)
    v_bot = gray[8::8, :].astype(np.float32)
    min_v = min(v_top.shape[0], v_bot.shape[0])
    v_diff = np.abs(v_top[:min_v, :] - v_bot[:min_v, :])
    block_score = float((h_diff.mean() + v_diff.mean()) / 2.0)

    # Double compression heuristic: periodic peaks in DCT histogram
    dct = cv2.dct(gray.astype(np.float32))
    dct_energy = float(np.mean(np.abs(dct)))

    artifact_score = min(100.0, block_score * 2.5 + max(0, dct_energy - 15) * 1.5)

    return {
        "block_boundary_score": round(block_score, 2),
        "dct_energy": round(dct_energy, 2),
        "compression_artifact_score": round(artifact_score, 1),
        "double_compression_likely": artifact_score > 45,
    }


def run_forensic_analysis(img_rgb: np.ndarray) -> dict:
    """Run complete forensic analysis pipeline."""
    fft = compute_fft_features(img_rgb)
    texture = compute_texture_features(img_rgb)
    noise = analyze_noise_patterns(img_rgb)
    compression = analyze_compression_artifacts(img_rgb)

    scores = [
        fft["fft_anomaly_score"] * 0.35,
        noise["noise_suspicion_score"] * 0.20,
        compression["compression_artifact_score"] * 0.20,
        min(100.0, texture["laplacian_variance"] / 10.0) * 0.25,
    ]
    suspicion_score = round(min(100.0, sum(scores)), 1)
    authenticity_score = round(100.0 - suspicion_score, 1)

    indicators = []
    if fft["fft_anomaly_score"] > 40:
        indicators.append("Elevated high-frequency energy in FFT spectrum")
    if noise["noise_suspicion_score"] > 35:
        indicators.append("Inconsistent noise distribution across image regions")
    if compression["compression_artifact_score"] > 40:
        indicators.append("Abnormal JPEG compression block patterns detected")
    if texture["laplacian_variance"] > 800:
        indicators.append("Unusually high texture sharpness variance")

    forensic_confidence = round(
        min(100.0, 50 + len(indicators) * 12 + suspicion_score * 0.2), 1
    )

    return {
        "fft_analysis": {
            "high_freq_energy": fft["high_freq_energy"],
            "fft_anomaly_score": fft["fft_anomaly_score"],
        },
        "fft_image_bytes": fft["fft_image_bytes"],
        "texture_analysis": texture,
        "noise_analysis": noise,
        "compression_analysis": compression,
        "authenticity_score": authenticity_score,
        "suspicion_score": suspicion_score,
        "forensic_confidence": forensic_confidence,
        "manipulation_indicators": indicators,
    }
