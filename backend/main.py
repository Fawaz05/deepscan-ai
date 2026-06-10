"""
DeepScan AI — FastAPI backend for Explainable Deepfake Detection.
"""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from api.v1.routes import router as v1_router
from config import API_PREFIX, CORS_ORIGINS, LOG_LEVEL, RATE_LIMIT
from services.model_loader import load_model

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL.upper(), logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

limiter = Limiter(key_func=get_remote_address, default_limits=[RATE_LIMIT])


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting DeepScan AI backend...")
    try:
        load_model()
        logger.info("Model pre-loaded successfully.")
    except Exception as exc:
        logger.error("Model pre-load failed (will retry on first request): %s", exc)
    yield
    logger.info("Shutting down DeepScan AI backend.")


app = FastAPI(
    title="DeepScan AI — Deepfake Detection API",
    description="Explainable deepfake image detection with Grad-CAM++, forensic analysis, and AI explanations",
    version="1.0.0",
    lifespan=lifespan,
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS if CORS_ORIGINS != ["*"] else ["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    duration_ms = (time.perf_counter() - start) * 1000
    logger.info("%s %s -> %s (%.1fms)", request.method, request.url.path, response.status_code, duration_ms)
    return response


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled error on %s %s", request.method, request.url.path)
    return JSONResponse(status_code=500, content={"detail": "Internal server error."})


app.include_router(v1_router, prefix=API_PREFIX)

# Legacy route aliases for backward compatibility
app.include_router(v1_router, tags=["legacy"])


@app.get("/")
async def root():
    return {
        "service": "DeepScan AI",
        "version": "1.0.0",
        "docs": "/docs",
        "health": f"{API_PREFIX}/health",
    }
