# Architecture

## Overview

DeepScan AI is a monorepo with a **Next.js** frontend and **FastAPI** backend orchestrating a multi-stage deepfake detection and explainability pipeline.

```
┌─────────────┐     REST API      ┌──────────────────────────────────────┐
│  Next.js    │ ────────────────► │  FastAPI Backend (api/v1)            │
│  Frontend   │                   │  ├── Upload / URL Analysis           │
│  (Vercel)   │ ◄──────────────── │  ├── Report Cache                    │
└─────────────┘     JSON+base64   │  └── Health / Download               │
                                  └──────────────┬───────────────────────┘
                                                 │
                    ┌────────────────────────────┼────────────────────────────┐
                    ▼                            ▼                            ▼
           ┌──────────────┐            ┌─────────────────┐          ┌──────────────┐
           │ Model Loader │            │ Inference Engine │          │ LLM Service  │
           │ MobileNetV2  │            │ Grad-CAM / CAM   │          │ Claude API   │
           └──────────────┘            │ Forensic Engine  │          │ + Fallback   │
                                        │ Social Safety    │          └──────────────┘
                                        └─────────────────┘
```

## Detection Model

| Property | Value |
|----------|-------|
| Backbone | MobileNetV2 (`mobilenetv2_1.00_224`) |
| Input | 224×224×3 RGB |
| Output | Sigmoid binary (0=real, 1=fake) |
| Format | Keras 3 directory (`config.json` + `model.weights.h5`) |

## Pipeline Stages

1. **Preprocessing** — bilateral denoise, resize, MobileNetV2 normalization
2. **Inference** — binary classification with confidence
3. **Explainability** — Grad-CAM++ with weight-based CAM fallback
4. **Region Extraction** — contour-based suspicious area detection
5. **Forensic Analysis** — FFT, noise, texture, compression artifacts
6. **Captioning** — BLIP (optional) or rule-based region descriptions
7. **Explanation** — evidence-based LLM or rule-based narrative
8. **Social Safety** — misuse risk scoring
9. **Report** — JSON cache with downloadable metadata

## Directory Structure

```
Deep/
├── backend/
│   ├── api/v1/          # Versioned API routes
│   ├── services/        # Model loader, inference, BLIP, LLM, social safety
│   ├── utils/           # Preprocessing, Grad-CAM, forensic, security
│   ├── models/          # Trained Keras model artifacts
│   └── main.py          # FastAPI entry point
├── frontend/            # Next.js 14 + TypeScript + Tailwind
├── tests/               # Unit and integration tests
└── docs/                # Documentation
```
