"""Pydantic request/response schemas for API v1."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, HttpUrl


class UrlRequest(BaseModel):
    url: HttpUrl


class HealthResponse(BaseModel):
    status: str
    service: str
    model_loaded: bool
    version: str


class ReportSummary(BaseModel):
    report_id: str | None = None
    prediction: str | None = None
    confidence: float | None = None
    trust_score: float | None = None
    risk_level: str | None = None
    timestamp: str | None = None


class ReportsListResponse(BaseModel):
    reports: list[ReportSummary]
    total: int


class AnalysisResponse(BaseModel):
    report_id: str
    prediction: str
    confidence: float
    trust_score: float
    risk_level: str
    forensic_explanation: str
    misuse_warning: str | None = None
    heatmap_image: str | None = None
    annotated_image: str | None = None
    fft_image: str | None = None
    suspicious_regions: list[Any] = Field(default_factory=list)
    region_captions: list[str] = Field(default_factory=list)
    fft_analysis: dict[str, Any] = Field(default_factory=dict)
    score_components: dict[str, Any] = Field(default_factory=dict)
    forensic_scores: dict[str, Any] = Field(default_factory=dict)
    social_safety: dict[str, Any] = Field(default_factory=dict)
    texture_analysis: dict[str, Any] = Field(default_factory=dict)
    noise_analysis: dict[str, Any] = Field(default_factory=dict)
    compression_analysis: dict[str, Any] = Field(default_factory=dict)
    timestamp: str
    model_info: dict[str, Any] = Field(default_factory=dict)
