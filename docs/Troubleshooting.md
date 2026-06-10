# Troubleshooting

## Model fails to load

**Symptom:** `Model path does not exist` or `Missing config.json`

**Fix:**
1. Verify `backend/models/model_finetuned_keras/` contains `config.json` and `model.weights.h5`
2. Set `MODEL_PATH` environment variable to the correct path
3. Extract `backend/model_finetuned.keras.zip` if directory is missing

## TensorFlow import errors

**Symptom:** `No module named 'tensorflow'`

**Fix:**
```bash
pip install tensorflow>=2.16.0
```

On Python 3.13, use TensorFlow 2.21+. For production, prefer Python 3.11.

## Grad-CAM heatmap is empty

The system uses a **weight-based CAM fallback** when gradient-based Grad-CAM++ is unavailable (common with Keras 3 nested models). Heatmaps should still be generated.

## BLIP captioning is slow

Set `USE_BLIP=false` (default). BLIP downloads ~1 GB on first use.

## CORS errors in browser

Set backend `CORS_ORIGINS` to include your frontend URL:
```
CORS_ORIGINS=https://your-app.vercel.app,http://localhost:3000
```

## Rate limiting (429)

Default: 30 requests/minute per IP. Adjust with `RATE_LIMIT` env var.

## Frontend cannot reach API

Verify `NEXT_PUBLIC_API_URL` points to the deployed backend (not localhost in production).

## imghdr removed (Python 3.13)

Fixed — validation now uses Pillow `Image.verify()`.
