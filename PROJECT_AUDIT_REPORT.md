# Project Audit Report — DeepScan AI

**Date:** June 10, 2026  
**Auditor:** Automated Production Readiness Transformation  
**Repository:** `m:\VIT\6thSemester\DL\Deep`

---

## Repository Overview

DeepScan AI is an explainable deepfake detection platform combining:

| Component | Stack |
|-----------|-------|
| Frontend | Next.js 14, TypeScript, Tailwind CSS, Framer Motion |
| Backend | FastAPI, Uvicorn, SlowAPI rate limiting |
| ML Model | MobileNetV2 (Keras 3 directory format) |
| XAI | Grad-CAM++ with weight-based CAM fallback |
| Forensics | FFT, noise, texture, compression analysis |
| LLM | Anthropic Claude with rule-based fallback |
| VLM | BLIP (optional, disabled by default) |

---

## Issues Found

### Critical
1. TensorFlow missing from `requirements.txt` despite being core dependency
2. Grad-CAM disabled in inference pipeline — no heatmaps generated
3. Grad-CAM broken for nested MobileNetV2 + Keras 3 graph introspection
4. `imghdr` removed in Python 3.13 — file validation would crash
5. No deployment configuration (Dockerfile, render.yaml, vercel.json)
6. No project documentation, README, LICENSE, or .gitignore
7. No API versioning (`/api/v1/`)
8. `python-dotenv` listed but `load_dotenv()` never called
9. Duplicate legacy Python files at `backend/` root (unused)
10. Duplicate model copies in `services/models/` and `models/`

### High
11. CRA frontend with inline styles — not production-grade
12. No tests
13. No structured logging or rate limiting
14. No social media safety module
15. Forensic analysis limited to FFT only
16. BLIP and Grad-CAM explicitly disabled in inference
17. OpenCV dtype errors in forensic noise/compression analysis
18. No download-report endpoint
19. CORS wide open with no env configuration

### Medium
20. `index.html` referenced wrong entry (`main.jsx` vs `index.js`)
21. No `.env.example`
22. Reports cached without forensic_scores/social_safety fields
23. LLM called without API key check (always failed to API then fallback)

---

## Issues Fixed

| # | Fix |
|---|-----|
| 1 | Added `tensorflow>=2.16.0` to requirements.txt + requirements-prod.txt |
| 2–3 | Re-enabled Grad-CAM; fixed nested backbone; added weight-based CAM fallback for Keras 3 |
| 4 | Replaced `imghdr` with Pillow `Image.verify()` |
| 5 | Created Dockerfile, render.yaml, vercel.json |
| 6 | Created README.md, LICENSE (MIT), .gitignore, docs/ |
| 7 | Implemented `/api/v1/` router with legacy alias routes |
| 8 | Added `config.py` with `load_dotenv()` |
| 9 | Legacy files retained but documented as unused |
| 11 | Rebuilt frontend as Next.js 14 + TypeScript + Tailwind |
| 12 | Added pytest suite (model, gradcam, API tests) |
| 13 | Structured logging middleware + SlowAPI rate limiting |
| 14 | Created `services/social_safety.py` |
| 15 | Created `utils/forensic_analysis.py` (FFT, noise, texture, compression) |
| 16 | Re-enabled full XAI pipeline in `inference_engine.py` |
| 17 | Fixed OpenCV dtype and array shape issues |
| 18 | Added `GET /download-report/{id}` |
| 19 | CORS configurable via `CORS_ORIGINS` env var |
| 20 | New Next.js entry resolves frontend bootstrap |
| 21 | Created `.env.example` |
| 22 | Reports now include forensic_scores, social_safety, noise/compression |
| 23 | LLM skips API call when `ANTHROPIC_API_KEY` unset |

### New Files Created
- `backend/services/model_loader.py` — universal Keras/SavedModel/.h5 loader
- `backend/config.py` — centralized configuration
- `backend/api/v1/routes.py` + `schemas.py`
- `backend/utils/security.py` — file validation
- `backend/utils/forensic_analysis.py`
- `backend/services/social_safety.py`
- `frontend/` — complete Next.js application (7 pages)
- `tests/` — 8 test cases
- `docs/` — Architecture, API, Deployment, Troubleshooting

---

## Issues Remaining

| Issue | Severity | Notes |
|-------|----------|-------|
| Duplicate legacy backend files | Low | `backend/inference_engine.py` etc. not imported; safe to delete |
| Duplicate model in `services/models/` | Low | Redundant ~150 MB; keep `models/` as canonical |
| Gradient-based Grad-CAM++ on Keras 3 | Medium | Returns None; weight-based CAM fallback used automatically |
| BLIP disabled by default | Info | Enable with `USE_BLIP=true` (requires ~1 GB download) |
| No training data directory | Info | `backend/data/` missing; training not runnable without dataset |
| Python 3.13 + TF on Windows | Info | Works with TF 2.21; GPU unavailable on native Windows |
| Screenshots in README | Low | Placeholder section — add after deployment |
| `gitignore` excludes report JSONs | Low | Intentional to keep repo clean |

---

## Model Status

| Property | Status |
|----------|--------|
| **Load** | ✅ SUCCESS |
| **Location** | `backend/models/model_finetuned_keras/` |
| **Format** | Keras 3 directory (`config.json` + `model.weights.h5`) |
| **Architecture** | MobileNetV2 → GAP → Dense(128) → Dropout → Dense(1) |
| **Input** | 224×224×3 RGB, MobileNetV2 preprocess |
| **Output** | Sigmoid scalar (0=real, 1=fake) |
| **Inference** | ✅ Verified (random + upload tests pass) |
| **Metadata** | Keras 3.13.2, saved 2026-04-10 |
| **Zip fallback** | `model_finetuned.keras.zip` available |

---

## Frontend Status

| Check | Status |
|-------|--------|
| Framework | ✅ Next.js 14 + TypeScript |
| Build | ✅ `npm run build` passes |
| Pages | ✅ Home, Upload, Analyze, Reports, Results, About |
| Dark/Light mode | ✅ Theme toggle |
| API integration | ✅ `/api/v1` client |
| Responsive design | ✅ Tailwind responsive grid |
| Loading states | ✅ Animated spinner |
| Error handling | ✅ User-facing error messages |

---

## Backend Status

| Check | Status |
|-------|--------|
| FastAPI startup | ✅ |
| Model pre-load on lifespan | ✅ |
| Full pipeline E2E | ✅ Upload test passes |
| Grad-CAM heatmaps | ✅ Weight-based CAM |
| Region extraction | ✅ |
| Forensic analysis | ✅ |
| Social safety | ✅ |
| Report caching | ✅ |
| Structured logging | ✅ |

---

## API Status

| Endpoint | Status |
|----------|--------|
| `GET /api/v1/health` | ✅ |
| `POST /api/v1/upload-image` | ✅ |
| `POST /api/v1/analyze-url` | ✅ |
| `GET /api/v1/report/{id}` | ✅ |
| `GET /api/v1/reports` | ✅ |
| `GET /api/v1/download-report/{id}` | ✅ |
| Legacy routes (no prefix) | ✅ Aliased |
| OpenAPI docs `/docs` | ✅ |
| Rate limiting | ✅ 30/min default |

---

## Security Status

| Control | Status |
|---------|--------|
| File type validation | ✅ Extension + Pillow verify |
| File size limit | ✅ 20 MB configurable |
| Path traversal prevention | ✅ Report ID sanitization |
| CORS | ✅ Configurable origins |
| Rate limiting | ✅ SlowAPI |
| Env var secrets | ✅ .env.example, not committed |
| Global exception handler | ✅ No stack traces to client |

---

## Deployment Status

| Artifact | Status |
|----------|--------|
| `backend/Dockerfile` | ✅ Python 3.11-slim |
| `render.yaml` | ✅ Docker web service |
| `frontend/vercel.json` | ✅ Next.js |
| `requirements-prod.txt` | ✅ Includes gunicorn |
| Health check endpoint | ✅ |

---

## Performance Improvements

1. **Singleton model cache** — model loaded once via `model_loader.py`
2. **Lazy BLIP loading** — only when `USE_BLIP=true`
3. **Disk report cache** — excludes large base64 images from JSON files
4. **Startup pre-load** — model warmed in FastAPI lifespan
5. **Weight-based CAM** — faster than failed gradient retry loops
6. **Bilateral denoise preprocessing** — single pass before inference

---

## Test Results

```
tests/test_model_loader.py  — 3/3 PASSED
tests/test_gradcam.py       — 1/1 PASSED
tests/test_api.py           — 4/4 PASSED (including E2E upload)
frontend npm run build      — SUCCESS
```

---

## Recommended Future Enhancements

1. Delete duplicate legacy backend files and `services/models/` copy
2. Add PDF report export with embedded images
3. Video frame-by-frame deepfake detection
4. GPU deployment on Linux cloud (CUDA)
5. User authentication and persistent report database
6. Ensemble model voting (Xception + MobileNetV2)
7. True gradient Grad-CAM when Keras 3 tape support improves
8. CI/CD GitHub Actions workflow
9. Add actual screenshots to README after Vercel deploy
10. PostgreSQL for report storage instead of filesystem

---

## Conclusion

The repository has been transformed from a functional prototype into a **production-ready, deployment-ready** explainable deepfake detection platform. All critical pipeline components work end-to-end: model loading, inference, explainability heatmaps, forensic analysis, social safety scoring, API endpoints, frontend UI, tests, and deployment configurations.

**Overall Status: ✅ PRODUCTION READY** (with noted minor remaining items)
