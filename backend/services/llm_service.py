"""
services/llm_service.py
-----------------------
LLM-based forensic explanation engine.
Uses Anthropic Claude (via API) to generate a coherent forensic report
from prediction + confidence + region captions + FFT analysis.
"""

import os
import anthropic


_client = None


def _get_client() -> anthropic.Anthropic:
    global _client
    if _client is None:
        api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        _client = anthropic.Anthropic(api_key=api_key)
    return _client


def generate_forensic_explanation(
    prediction: str,
    confidence: float,
    trust_score: float,
    risk_level: str,
    region_captions: list[str],
    fft_anomaly_score: float,
    high_freq_energy: float,
    num_regions: int,
    mean_heatmap_intensity: float,
    manipulation_indicators: list[str] | None = None,
    authenticity_score: float | None = None,
) -> str:
    """
    Generate a coherent forensic narrative using an LLM.

    Returns a 3–5 sentence human-readable explanation.
    """
    client = _get_client()

    captions_text = "\n".join(
        f"  - Region {i+1}: {cap}" for i, cap in enumerate(region_captions)
    ) if region_captions else "  - No specific regions detected."

    indicators_text = "\n".join(
        f"  - {ind}" for ind in (manipulation_indicators or [])
    ) or "  - None detected."

    auth_text = f"{authenticity_score:.1f}/100" if authenticity_score is not None else "N/A"

    prompt = f"""You are an expert digital forensics analyst specialising in deepfake detection.
Analyse the following detection results and produce a concise forensic explanation (3–5 sentences).
Focus on the technical evidence. Be specific, authoritative, and clear. Do NOT start with "I".

Detection Results:
- Verdict: {prediction.upper()}
- Model confidence: {confidence*100:.1f}%
- Trust Score: {trust_score}/100
- Risk Level: {risk_level}
- Suspicious regions detected: {num_regions}
- Mean activation intensity: {mean_heatmap_intensity:.2f}
- FFT high-frequency energy ratio: {high_freq_energy:.3f}
- FFT anomaly score: {fft_anomaly_score:.1f}/100
- Authenticity score: {auth_text}

Forensic Indicators (use ONLY these — do not invent new evidence):
{indicators_text}

Region Analysis Captions:
{captions_text}

Write a forensic explanation that:
1. States the verdict clearly with confidence
2. Cites specific visual evidence from the region captions
3. References the frequency-domain analysis if anomalous
4. Mentions what specific manipulation artefacts suggest
5. Ends with a risk assessment sentence

Return ONLY the explanation paragraph, no preamble."""

    if not os.environ.get("ANTHROPIC_API_KEY"):
        return _fallback_explanation(
            prediction,
            confidence,
            region_captions,
            fft_anomaly_score,
            risk_level,
            manipulation_indicators or [],
            mean_heatmap_intensity,
            num_regions,
        )

    try:
        message = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=300,
            messages=[{"role": "user", "content": prompt}],
        )
        return message.content[0].text.strip()
    except Exception:
        return _fallback_explanation(
            prediction,
            confidence,
            region_captions,
            fft_anomaly_score,
            risk_level,
            manipulation_indicators or [],
            mean_heatmap_intensity,
            num_regions,
        )


def _fallback_explanation(
    prediction: str,
    confidence: float,
    captions: list[str],
    fft_anomaly: float,
    risk_level: str,
    indicators: list[str] | None = None,
    mean_heatmap_intensity: float = 0.0,
    num_regions: int = 0,
) -> str:
    """Rule-based fallback when LLM API is unavailable."""
    verdict_str = (
        f"This image is classified as {prediction.upper()} with "
        f"{confidence*100:.1f}% confidence."
    )

    evidence_parts = []
    if num_regions > 0 and mean_heatmap_intensity > 0.3:
        evidence_parts.append(
            f"Grad-CAM++ analysis identified {num_regions} suspicious region(s) "
            f"with mean activation intensity of {mean_heatmap_intensity:.2f}"
        )
    if captions:
        evidence_parts.append(
            f"Visual analysis of suspicious regions reveals {captions[0].lower()}"
        )
    for ind in (indicators or [])[:2]:
        evidence_parts.append(ind)
    if fft_anomaly > 40:
        evidence_parts.append(
            f"Frequency-domain analysis detects elevated high-frequency energy "
            f"(anomaly score: {fft_anomaly:.0f}/100), a characteristic signature "
            f"of GAN-generated imagery"
        )

    evidence_str = (
        ". ".join(evidence_parts) + "." if evidence_parts
        else "Multiple forensic indicators were identified."
    )

    risk_str = {
        "High":   "This image poses a HIGH risk of misuse in misinformation or impersonation campaigns.",
        "Medium": "This image presents a MEDIUM risk and warrants further scrutiny.",
        "Low":    "Risk classification is LOW; the image appears authentic.",
    }.get(risk_level, "")

    return f"{verdict_str} {evidence_str} {risk_str}"
