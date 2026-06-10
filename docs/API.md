# API Documentation

Base URL: `http://localhost:8000/api/v1`

Interactive docs: `http://localhost:8000/docs`

## Endpoints

### `GET /health`

Health check with model load status.

**Response:**
```json
{
  "status": "ok",
  "service": "deepfake-detection-api",
  "model_loaded": true,
  "version": "1.0.0"
}
```

### `POST /upload-image`

Upload an image for forensic analysis.

**Request:** `multipart/form-data` with `file` field (image, max 20 MB)

**Response:** Full analysis report including prediction, heatmap (base64), forensic scores, and social safety assessment.

### `POST /analyze-url`

Analyze an image from a URL.

**Request:**
```json
{ "url": "https://example.com/image.jpg" }
```

### `GET /report/{report_id}`

Retrieve cached report metadata (images excluded from disk cache).

### `GET /reports?limit=20`

List recent reports.

### `GET /download-report/{report_id}`

Download report JSON file.

## Legacy Routes

The same endpoints are also available without the `/api/v1` prefix for backward compatibility.

## Error Codes

| Code | Meaning |
|------|---------|
| 400 | Invalid input / file type |
| 413 | File too large |
| 422 | Image decode or URL fetch failed |
| 429 | Rate limit exceeded |
| 500 | Pipeline error |
