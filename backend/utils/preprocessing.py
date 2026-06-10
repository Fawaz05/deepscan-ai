"""
utils/preprocessing.py
-----------------------
Image preprocessing pipeline including:
- Standard MobileNetV2 preprocessing
- FFT frequency-domain analysis
- Noise robustness preprocessing
- Image loading from file/URL
"""

import io
import numpy as np
import cv2
import requests
from PIL import Image
import tensorflow as tf
from keras import applications


IMG_SIZE = (224, 224)


# ─────────────────────────────────────────────
# Core preprocessing
# ─────────────────────────────────────────────

def load_image_from_bytes(data: bytes) -> np.ndarray:
    """Decode image bytes to RGB numpy array."""
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image bytes.")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def load_image_from_url(url: str) -> np.ndarray:
    """Fetch image from URL and return RGB numpy array."""
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 Chrome/120.0 Safari/537.36"
        )
    }
    response = requests.get(url, headers=headers, timeout=15)
    response.raise_for_status()
    return load_image_from_bytes(response.content)


def preprocess_for_model(img_rgb: np.ndarray) -> np.ndarray:
    """
    Resize to 224×224, apply MobileNetV2 preprocess_input (-1 to 1),
    add batch dimension.
    """
    resized = cv2.resize(img_rgb, IMG_SIZE, interpolation=cv2.INTER_LANCZOS4)
    tensor = resized.astype(np.float32)
    tensor = applications.mobilenet_v2.preprocess_input(tensor)
    return np.expand_dims(tensor, axis=0)   # (1, 224, 224, 3)


# ─────────────────────────────────────────────
# Noise robustness preprocessing
# ─────────────────────────────────────────────

def denoise_preprocessing(img_rgb: np.ndarray) -> np.ndarray:
    """
    Apply gentle bilateral filtering to reduce JPEG compression artifacts
    while preserving edges — improves consistency for manipulated images.
    """
    denoised = cv2.bilateralFilter(img_rgb, d=5, sigmaColor=20, sigmaSpace=20)
    return denoised


def preprocess_with_noise_robustness(img_rgb: np.ndarray) -> np.ndarray:
    """Full pipeline: denoise → resize → normalise → batch."""
    clean = denoise_preprocessing(img_rgb)
    return preprocess_for_model(clean)


# ─────────────────────────────────────────────
# FFT frequency analysis
# ─────────────────────────────────────────────

def compute_fft_features(img_rgb: np.ndarray) -> dict:
    """
    Compute FFT-based frequency spectrum features.

    Deepfakes often exhibit characteristic high-frequency artefacts
    (aliasing from upsampling, GAN ringing etc.) that show up as
    anomalous energy in the frequency domain.

    Returns:
        fft_magnitude_map  – 2-D log magnitude spectrum (224×224)
        high_freq_energy   – ratio of energy in high-frequency band
        fft_anomaly_score  – 0-100 score; higher → more suspicious
        fft_image_bytes    – PNG bytes of the spectrum visualisation
    """
    # Work in grayscale
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    gray_resized = cv2.resize(gray, IMG_SIZE)

    # 2-D FFT
    f = np.fft.fft2(gray_resized.astype(np.float32))
    fshift = np.fft.fftshift(f)

    magnitude = np.abs(fshift)
    log_mag = np.log1p(magnitude)

    # Normalise for visualisation
    norm_mag = cv2.normalize(log_mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    fft_colored = cv2.applyColorMap(norm_mag, cv2.COLORMAP_JET)
    _, fft_png = cv2.imencode(".png", fft_colored)
    fft_image_bytes = fft_png.tobytes()

    # High-frequency energy ratio
    h, w = IMG_SIZE
    cy, cx = h // 2, w // 2
    r_low  = min(h, w) // 8   # ~low-freq inner circle radius

    mask_low  = np.zeros((h, w), dtype=bool)
    mask_low[cy - r_low:cy + r_low, cx - r_low:cx + r_low] = True

    total_energy  = float(np.sum(magnitude ** 2))
    low_energy    = float(np.sum((magnitude * mask_low) ** 2))
    high_energy   = total_energy - low_energy
    hf_ratio      = high_energy / (total_energy + 1e-8)

    # Anomaly heuristic: deepfakes tend to have elevated mid-high freq energy
    # Typical natural images: hf_ratio ~ 0.70–0.85
    # GAN images: hf_ratio can reach 0.90+
    baseline = 0.78
    deviation = max(0.0, hf_ratio - baseline) / (1.0 - baseline)
    fft_anomaly_score = min(100.0, deviation * 200.0)

    return {
        "fft_magnitude_map": norm_mag,          # (224,224) uint8
        "high_freq_energy":  round(hf_ratio, 4),
        "fft_anomaly_score": round(fft_anomaly_score, 1),
        "fft_image_bytes":   fft_image_bytes,
    }


# ─────────────────────────────────────────────
# Channel statistics (texture complexity)
# ─────────────────────────────────────────────

def compute_texture_features(img_rgb: np.ndarray) -> dict:
    """
    Laplacian variance (focus/sharpness measure) and channel statistics.
    Inconsistent sharpness across facial regions is a deepfake tell.
    """
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    # Per-channel stats
    stats = {}
    for i, ch in enumerate(["r", "g", "b"]):
        channel = img_rgb[:, :, i].astype(np.float32)
        stats[f"{ch}_mean"] = round(float(channel.mean()), 2)
        stats[f"{ch}_std"]  = round(float(channel.std()),  2)

    return {
        "laplacian_variance": round(lap_var, 2),
        "channel_stats": stats,
    }
