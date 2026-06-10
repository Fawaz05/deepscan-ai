FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-prod.txt .
RUN pip install --no-cache-dir -r requirements-prod.txt

COPY . .

ENV MODEL_PATH=/app/models/model_finetuned_keras
ENV REPORTS_DIR=/app/reports
ENV USE_GRADCAM=true
ENV USE_BLIP=false
ENV PRELOAD_MODEL=false
ENV TF_CPP_MIN_LOG_LEVEL=2
ENV PYTHONUNBUFFERED=1

# Render injects PORT at runtime (defaults to 8000 locally)
EXPOSE 8000

CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"]
