"""API v1 route handlers."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse

from api.v1.schemas import AnalysisResponse, HealthResponse, ReportsListResponse, UrlRequest
from config import REPORTS_DIR, UPLOAD_MAX_BYTES
from services.inference_engine import analyze_image, get_model
from utils.preprocessing import load_image_from_bytes, load_image_from_url
from utils.security import validate_image_bytes, validate_image_upload

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health():
    model_loaded = False
    try:
        get_model()
        model_loaded = True
    except Exception as exc:
        logger.warning("Model not loaded at health check: %s", exc)

    return HealthResponse(
        status="ok",
        service="deepfake-detection-api",
        model_loaded=model_loaded,
        version="1.0.0",
    )


@router.post("/upload-image", response_model=AnalysisResponse)
async def upload_image(request: Request, file: UploadFile = File(...)):
    validate_image_upload(file)

    data = await file.read()
    if len(data) > UPLOAD_MAX_BYTES:
        raise HTTPException(413, detail=f"Image too large. Maximum {UPLOAD_MAX_BYTES // (1024*1024)} MB.")

    validate_image_bytes(data)

    try:
        img_rgb = load_image_from_bytes(data)
    except Exception as exc:
        raise HTTPException(422, detail=f"Could not decode image: {exc}") from exc

    try:
        report = analyze_image(img_rgb)
    except Exception as exc:
        logger.exception("Analysis pipeline error")
        raise HTTPException(500, detail=f"Analysis pipeline error: {exc}") from exc

    return JSONResponse(content=report)


@router.post("/analyze-url", response_model=AnalysisResponse)
async def analyze_url(body: UrlRequest):
    url = str(body.url).strip()
    if not url.startswith(("http://", "https://")):
        raise HTTPException(400, detail="URL must start with http:// or https://")

    try:
        img_rgb = load_image_from_url(url)
    except Exception as exc:
        raise HTTPException(422, detail=f"Could not fetch image from URL: {exc}") from exc

    try:
        report = analyze_image(img_rgb)
    except Exception as exc:
        logger.exception("Analysis pipeline error")
        raise HTTPException(500, detail=f"Analysis pipeline error: {exc}") from exc

    return JSONResponse(content=report)


@router.get("/report/{report_id}")
async def get_report(report_id: str):
    if ".." in report_id or "/" in report_id or "\\" in report_id:
        raise HTTPException(400, detail="Invalid report ID.")

    report_path = Path(REPORTS_DIR) / f"{report_id}.json"
    if not report_path.exists():
        raise HTTPException(404, detail=f"Report '{report_id}' not found.")

    with open(report_path, encoding="utf-8") as f:
        report = json.load(f)
    return JSONResponse(content=report)


@router.get("/reports", response_model=ReportsListResponse)
async def list_reports(limit: int = 20):
    limit = max(1, min(limit, 100))
    reports_dir = Path(REPORTS_DIR)
    files = sorted(reports_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    results = []
    for fp in files[:limit]:
        with open(fp, encoding="utf-8") as f:
            data = json.load(f)
        results.append({
            "report_id": data.get("report_id"),
            "prediction": data.get("prediction"),
            "confidence": data.get("confidence"),
            "trust_score": data.get("trust_score"),
            "risk_level": data.get("risk_level"),
            "timestamp": data.get("timestamp"),
        })
    return {"reports": results, "total": len(results)}


@router.get("/download-report/{report_id}")
async def download_report(report_id: str):
    if ".." in report_id or "/" in report_id or "\\" in report_id:
        raise HTTPException(400, detail="Invalid report ID.")

    report_path = Path(REPORTS_DIR) / f"{report_id}.json"
    if not report_path.exists():
        raise HTTPException(404, detail=f"Report '{report_id}' not found.")

    return FileResponse(
        path=str(report_path),
        media_type="application/json",
        filename=f"deepfake-report-{report_id}.json",
    )
