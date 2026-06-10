# DeepScan AI — Explainable Deepfake Detection Platform

Production-ready explainable deepfake detection platform with Grad-CAM++ heatmaps, forensic frequency analysis, AI-powered explanations, and social media safety risk assessment.

![Python](https://img.shields.io/badge/Python-3.10--3.13-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.16+-orange)
![Next.js](https://img.shields.io/badge/Next.js-14-black)
![FastAPI](https://img.shields.io/badge/FastAPI-0.111+-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

## Features

- **Deepfake Detection** — MobileNetV2 fine-tuned classifier (real vs fake)
- **Explainable AI** — Grad-CAM++ with weight-based CAM fallback, region extraction
- **Forensic Analysis** — FFT spectrum, noise uniformity, texture, JPEG compression artifacts
- **AI Explanations** — Claude-powered forensic narratives with evidence-based fallback
- **Social Safety** — Cyberbullying, misinformation, impersonation, identity abuse risk scoring
- **Vision-Language** — Optional BLIP region captioning
- **Report System** — Cached JSON reports with download support
- **Production API** — Versioned `/api/v1/`, rate limiting, CORS, structured logging

## Architecture

```
Frontend (Next.js/Vercel) ──► FastAPI Backend (Render/Railway) ──► MobileNetV2 Model
                                      │
                                      ├── Grad-CAM++ / CAM Heatmaps
                                      ├── Forensic Engine (FFT, noise, compression)
                                      ├── BLIP Captioning (optional)
                                      └── Claude LLM Explanations (optional)
```

See [docs/Architecture.md](docs/Architecture.md) for details.

## Technology Stack

| Layer | Technologies |
|-------|-------------|
| Frontend | Next.js 14, TypeScript, Tailwind CSS, Framer Motion |
| Backend | FastAPI, Uvicorn, Pydantic, SlowAPI |
| ML | TensorFlow/Keras 3, MobileNetV2, OpenCV, NumPy |
| XAI | Grad-CAM++, weight-based CAM, contour region extraction |
| VLM | HuggingFace BLIP (optional) |
| LLM | Anthropic Claude (optional) |

## Installation

### Prerequisites

- Python 3.10+ (3.11 recommended for TensorFlow)
- Node.js 18+
- 4 GB+ RAM (model loading)

### Backend

```bash
cd backend
python -m venv venv
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

pip install -r requirements.txt
cp ../.env.example ../.env   # edit as needed
uvicorn main:app --reload --port 8000
```

### Frontend

```bash
cd frontend
npm install
cp ../.env.example .env.local   # set NEXT_PUBLIC_API_URL
npm run dev
```

Open **http://localhost:3000**

## Usage

1. **Upload** — Drag and drop an image on `/upload`
2. **URL Analysis** — Paste an image URL on `/analyze`
3. **View Results** — Prediction, trust score, heatmaps, forensic scores, AI explanation
4. **Reports** — Browse past analyses on `/reports`

## API Documentation

Base URL: `http://localhost:8000/api/v1`

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check |
| POST | `/upload-image` | Upload image for analysis |
| POST | `/analyze-url` | Analyze image from URL |
| GET | `/report/{id}` | Get cached report |
| GET | `/reports` | List recent reports |
| GET | `/download-report/{id}` | Download report JSON |

Full docs: [docs/API.md](docs/API.md) | Interactive: `http://localhost:8000/docs`

## Folder Structure

```
Deep/
├── backend/
│   ├── api/v1/              # Versioned API routes
│   ├── services/            # Inference, model loader, BLIP, LLM
│   ├── utils/               # Preprocessing, Grad-CAM, forensic, security
│   ├── models/              # Trained Keras model
│   ├── main.py              # FastAPI entry
│   ├── Dockerfile           # Container build
│   └── requirements.txt
├── frontend/                # Next.js application
├── tests/                   # pytest suite
├── docs/                    # Documentation
├── render.yaml              # Render deployment
└── vercel.json              # Vercel deployment
```

## Deployment

| Service | Platform | Config |
|---------|----------|--------|
| Frontend | Vercel | Root: `frontend`, env: `NEXT_PUBLIC_API_URL` |
| Backend | Render | `render.yaml` + Docker |
| Backend (alt) | Railway | `backend/Dockerfile` |

See [docs/Deployment.md](docs/Deployment.md)

## Environment Variables

Copy `.env.example` to `.env` and configure:

```env
MODEL_PATH=backend/models/model_finetuned_keras
USE_GRADCAM=true
USE_BLIP=false
ANTHROPIC_API_KEY=
CORS_ORIGINS=http://localhost:3000
NEXT_PUBLIC_API_URL=http://localhost:8000
```

## Screenshots

> Add screenshots of the Upload page, Results view, and Heatmap overlay after deployment.

## Testing

```bash
cd backend && pip install -r requirements.txt
cd .. && python -m pytest tests/ -v -m "not slow"
```

## Troubleshooting

See [docs/Troubleshooting.md](docs/Troubleshooting.md)

## Future Improvements

- Video deepfake detection pipeline
- Ensemble models (Xception, EfficientNet)
- Real-time webcam analysis
- PDF report export with embedded heatmaps
- User authentication and report history
- GPU inference on cloud (CUDA)

## License

MIT License — see [LICENSE](LICENSE)
