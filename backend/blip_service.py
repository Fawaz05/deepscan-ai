"""
services/blip_service.py
------------------------
Vision-Language reasoning using Salesforce BLIP.
Generates anomaly captions for suspicious image regions.
"""

import io
import numpy as np
from PIL import Image

# Lazy import — only loaded when first used
_processor = None
_model     = None


def _load_blip():
    global _processor, _model
    if _processor is None:
        from transformers import BlipProcessor, BlipForConditionalGeneration
        import torch

        print("[BLIP] Loading model... (first call only)")
        model_name = "Salesforce/blip-image-captioning-large"
        _processor = BlipProcessor.from_pretrained(model_name)
        _model     = BlipForConditionalGeneration.from_pretrained(
            model_name, torch_dtype=torch.float16
        )
        device = "cuda" if __import__("torch").cuda.is_available() else "cpu"
        _model = _model.to(device)
        _model.eval()
        print(f"[BLIP] Loaded on {device}")


def caption_region(crop_rgb: np.ndarray) -> str:
    """
    Generate an anomaly-focused caption for a suspicious image crop.

    Args:
        crop_rgb: RGB numpy array of the suspicious region.

    Returns:
        Human-readable caption describing visual inconsistencies.
    """
    _load_blip()

    import torch

    pil_img = Image.fromarray(crop_rgb.astype(np.uint8))

    # Conditional captioning with forensic prompt
    prompt = (
        "Analyze this region and describe any visual inconsistencies, "
        "artifacts, or unnatural patterns that suggest image manipulation:"
    )

    inputs = _processor(
        images=pil_img,
        text=prompt,
        return_tensors="pt",
    ).to(_model.device, torch.float16)

    with torch.no_grad():
        output_ids = _model.generate(
            **inputs,
            max_new_tokens=80,
            num_beams=5,
            early_stopping=True,
            temperature=0.7,
        )

    caption = _processor.decode(output_ids[0], skip_special_tokens=True)

    # Strip repeated prompt if model echoes it
    if caption.lower().startswith(prompt[:30].lower()):
        caption = caption[len(prompt):].strip(" :")

    return caption if caption else "Region shows potential visual inconsistencies."


def caption_multiple_regions(crops: list[np.ndarray]) -> list[str]:
    """Generate captions for multiple regions."""
    return [caption_region(crop) for crop in crops]


# ─────────────────────────────────────────────
# Fallback: rule-based caption when BLIP unavailable
# ─────────────────────────────────────────────

def rule_based_caption(region: dict) -> str:
    """
    Generate a descriptive caption using heuristics when BLIP is not available.
    """
    intensity = region.get("intensity", 0)
    x, y, bw, bh = region.get("bbox", (0, 0, 0, 0))

    position_desc = ""
    if y < 100:
        position_desc = "upper facial region"
    elif y > 200:
        position_desc = "lower facial region"
    else:
        position_desc = "mid-facial area"

    if intensity > 0.75:
        severity = "strong evidence of manipulation with"
    elif intensity > 0.5:
        severity = "notable inconsistencies including"
    else:
        severity = "subtle artefacts suggesting"

    if bw > 80 and bh > 80:
        scope = "a large area showing"
    else:
        scope = "a localised region with"

    return (
        f"The {position_desc} exhibits {severity} texture distortion and "
        f"unnatural blending artefacts. {scope.capitalize()} "
        f"anomalous gradient patterns consistent with GAN synthesis."
    )
