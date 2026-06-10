"""
Social media safety risk assessment module.

Evaluates cyberbullying, misinformation, impersonation, and identity abuse risks.
"""

from __future__ import annotations


def assess_social_risks(
    prediction: str,
    confidence: float,
    trust_score: float,
    suspicion_score: float,
    num_regions: int,
) -> dict:
    """
    Compute social media safety categories and warnings.

    Categories: Safe | Suspicious | High Risk
    """
    is_fake = prediction.lower() == "fake"
    conf_pct = confidence * 100 if confidence <= 1 else confidence

    # Base risk from detection
    if is_fake and conf_pct >= 70:
        base_risk = "High Risk"
    elif is_fake or suspicion_score >= 55:
        base_risk = "Suspicious"
    else:
        base_risk = "Safe"

    cyberbullying = min(100.0, (conf_pct if is_fake else 0) * 0.6 + suspicion_score * 0.3)
    misinformation = min(100.0, (conf_pct if is_fake else 0) * 0.7 + (100 - trust_score) * 0.2)
    impersonation = min(100.0, (conf_pct if is_fake else 0) * 0.8 + num_regions * 5)
    identity_abuse = min(100.0, (conf_pct if is_fake else 0) * 0.5 + suspicion_score * 0.4)

    warnings = []
    if is_fake and conf_pct >= 60:
        warnings.append(
            "This image may be used to harass or target individuals through manipulated content."
        )
    if is_fake and conf_pct >= 50:
        warnings.append(
            "Manipulated media detected — verify source before sharing to prevent misinformation spread."
        )
    if impersonation >= 50:
        warnings.append(
            "High impersonation risk: facial manipulation may be used to falsely represent someone."
        )
    if identity_abuse >= 45:
        warnings.append(
            "Potential identity abuse: synthetic facial features may violate personal likeness rights."
        )

    return {
        "safety_category": base_risk,
        "cyberbullying_risk": round(cyberbullying, 1),
        "misinformation_risk": round(misinformation, 1),
        "impersonation_risk": round(impersonation, 1),
        "identity_abuse_risk": round(identity_abuse, 1),
        "warnings": warnings,
    }
