"""
Application configuration loaded from environment variables.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent

# Model
MODEL_PATH = os.environ.get("MODEL_PATH", str(BASE_DIR / "models" / "model_finetuned_keras"))
USE_BLIP = os.environ.get("USE_BLIP", "false").lower() == "true"
USE_GRADCAM = os.environ.get("USE_GRADCAM", "true").lower() == "true"
PRELOAD_MODEL = os.environ.get("PRELOAD_MODEL", "true").lower() == "true"

# Storage
REPORTS_DIR = os.environ.get("REPORTS_DIR", str(BASE_DIR / "reports"))
UPLOAD_MAX_BYTES = int(os.environ.get("UPLOAD_MAX_BYTES", str(20 * 1024 * 1024)))

# API
API_PREFIX = "/api/v1"
CORS_ORIGINS = [
    o.strip()
    for o in os.environ.get("CORS_ORIGINS", "http://localhost:3000,http://localhost:5173").split(",")
    if o.strip()
]
RATE_LIMIT = os.environ.get("RATE_LIMIT", "30/minute")

# LLM
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")

# Logging
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO")

# Ensure directories exist
Path(REPORTS_DIR).mkdir(parents=True, exist_ok=True)
