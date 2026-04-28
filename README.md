# News Classification API

Industry-standard FastAPI service for Vietnamese news classification.

## Architecture

```
app/
├── api/v1/endpoints/news.py   ← REST endpoints (classify, batch, health)
├── core/config.py             ← Pydantic Settings (env-driven)
├── core/logging.py            ← Structured logging (structlog / stdlib)
├── ml/model_handler.py        ← Singleton artifact loader + inference
├── ml/processor.py            ← Text cleaning & tokenisation pipeline
├── schemas/news.py            ← Pydantic request/response models
├── services/classification.py ← Orchestration & business logic
└── main.py                    ← FastAPI app + lifespan context
```

## Quick Start

### 1. Create & activate virtualenv

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env — set MODEL_NAME, VECTORIZER_NAME, LOG_FORMAT, etc.
```

### 3. Add trained model artifacts

Place the following files in the `models/` directory:

| File | Description |
|---|---|
| `news_classifier.pkl` | Trained sklearn classifier (joblib) |
| `tfidf_vectorizer.pkl` | Fitted TfidfVectorizer (joblib) |
| `label_encoder.pkl` | LabelEncoder (optional, joblib) |

### 4. Run locally

```bash
uvicorn app.main:app --reload --port 8000
```

Open: <http://localhost:8000/docs>

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/api/v1/news/health` | Health check & model status |
| `GET` | `/api/v1/news/model/info` | Class names, threshold, version |
| `POST` | `/api/v1/news/classify` | Classify a single article |
| `POST` | `/api/v1/news/classify/batch` | Classify up to 32 articles |

### Example request

```bash
curl -X POST http://localhost:8000/api/v1/news/classify \
  -H "Content-Type: application/json" \
  -d '{"title": "Bóng đá Việt Nam", "content": "Đội tuyển Việt Nam giành chiến thắng ấn tượng."}'
```

### Example response

```json
{
  "status": "success",
  "model_version": "news_classifier",
  "result": {
    "label": "the_thao",
    "confidence": 0.923,
    "probabilities": {
      "the_thao": 0.923,
      "kinh_te": 0.031,
      "chinh_tri": 0.046
    },
    "below_threshold": false
  }
}
```

---

## Running Tests

```bash
pytest -v
```

Tests use dummy model stubs — no real artifacts required.

---

## Docker

```bash
# Build & run
docker compose up --build

# Health check
curl http://localhost:8000/api/v1/news/health
```

---

## Configuration Reference

All options are read from `.env` (see `.env.example`):

| Key | Default | Description |
|-----|---------|-------------|
| `MODEL_DIR` | `models` | Directory containing trained artifacts |
| `MODEL_NAME` | `news_classifier.pkl` | Classifier artifact filename |
| `VECTORIZER_NAME` | `tfidf_vectorizer.pkl` | Vectorizer artifact filename |
| `LABEL_ENCODER_NAME` | `label_encoder.pkl` | Label encoder (optional) |
| `CONFIDENCE_THRESHOLD` | `0.5` | Minimum confidence to flag result as reliable |
| `MAX_TEXT_LENGTH` | `10000` | Maximum input characters |
| `LOG_LEVEL` | `INFO` | Python log level |
| `LOG_FORMAT` | `json` | `json` (production) or `text` (development) |