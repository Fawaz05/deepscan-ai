# Deployment Guide

## Architecture

| Component | Platform | Notes |
|-----------|----------|-------|
| Frontend | **Vercel** | Next.js 14, set root directory to `frontend` |
| Backend | **Render** or **Railway** | Docker-based, includes TensorFlow |

## Backend — Render

1. Connect GitHub repository to Render
2. Use `render.yaml` (Docker runtime)
3. Set environment variables:
   - `CORS_ORIGINS` → your Vercel frontend URL
   - `ANTHROPIC_API_KEY` (optional)
   - `USE_BLIP=false` (recommended for starter plan)
4. Deploy — first boot may take 2–3 minutes for model load

## Backend — Docker (local / Railway)

```bash
cd backend
docker build -t deepscan-api .
docker run -p 8000:8000 -e CORS_ORIGINS=http://localhost:3000 deepscan-api
```

## Frontend — Vercel

1. Import repository on Vercel
2. Set **Root Directory** to `frontend`
3. Framework preset: **Next.js**
4. Environment variable:
   - `NEXT_PUBLIC_API_URL` → your Render/Railway backend URL
5. Deploy

## Python Version

- **Recommended:** Python 3.10 or 3.11 for maximum TensorFlow compatibility
- **Supported:** Python 3.13 with TensorFlow 2.21+ (tested)

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r backend/requirements.txt
cd backend && uvicorn main:app --host 0.0.0.0 --port 8000
```

## Local Development

```bash
# Terminal 1 — Backend
cd backend
python -m venv venv && venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --reload --port 8000

# Terminal 2 — Frontend
cd frontend
npm install
npm run dev
```

Open `http://localhost:3000`
