from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache
from typing import List


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── App ──────────────────────────────────────────────────────────────
    APP_NAME: str = "BBC News Classifier API"
    APP_VERSION: str = "1.0.0"
    APP_DESCRIPTION: str = (
        "Industry-standard FastAPI service for BBC News classification. "
        "Supports TF-IDF (sklearn) and BERT (HuggingFace) backends."
    )
    DEBUG: bool = False

    # ── API ──────────────────────────────────────────────────────────────
    API_PREFIX: str = "/api/v1"
    ALLOWED_HOSTS: List[str] = ["*"]

    # ── Active models ─────────────────────────────────────────────────────
    # Comma-separated list of model types to load at startup.
    # Supported values: "tfidf", "bert"
    # Examples:
    #   ACTIVE_MODELS=tfidf           → load only TF-IDF
    #   ACTIVE_MODELS=bert            → load only BERT
    #   ACTIVE_MODELS=tfidf,bert      → load both (requires more RAM/GPU)
    ACTIVE_MODELS: str = "tfidf"

    @property
    def active_model_list(self) -> List[str]:
        """Parse ACTIVE_MODELS into a clean list of model-type strings."""
        return [m.strip().lower() for m in self.ACTIVE_MODELS.split(",") if m.strip()]

    # ── Common inference settings ─────────────────────────────────────────
    MAX_TEXT_LENGTH: int = 10_000          # max chars per request (safety cap)
    CONFIDENCE_THRESHOLD: float = 0.5     # flag results below this

    # ── TF-IDF model artifacts ────────────────────────────────────────────
    # Place trained joblib artifacts in TFIDF_MODEL_DIR:
    #   {TFIDF_MODEL_DIR}/{TFIDF_MODEL_NAME}
    #   {TFIDF_MODEL_DIR}/{TFIDF_VECTORIZER_NAME}
    #   {TFIDF_MODEL_DIR}/{TFIDF_LABEL_ENCODER_NAME}   (optional)
    TFIDF_MODEL_DIR: str = "models/tfidf"
    TFIDF_MODEL_NAME: str = "news_classifier.pkl"
    TFIDF_VECTORIZER_NAME: str = "tfidf_vectorizer.pkl"
    TFIDF_LABEL_ENCODER_NAME: str = "label_encoder.pkl"

    # ── BERT model ────────────────────────────────────────────────────────
    # Option A — local fine-tuned checkpoint:
    #   BERT_MODEL_DIR = models/bert
    #   BERT_MODEL_NAME = bbc-bert-finetuned   (directory name inside BERT_MODEL_DIR)
    #
    # Option B — HuggingFace Hub model ID (requires internet on first run):
    #   BERT_MODEL_DIR = models/bert           (must NOT contain BERT_MODEL_NAME)
    #   BERT_MODEL_NAME = bert-base-uncased    (HF hub id)
    BERT_MODEL_DIR: str = "models/bert"
    BERT_MODEL_NAME: str = "bert-base-uncased"   # HF hub id or local sub-dir name
    BERT_MAX_LENGTH: int = 512
    BERT_DEVICE: str = "cpu"                     # "cpu" | "cuda" | "mps"

    # ── Logging ───────────────────────────────────────────────────────────
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "json"   # "json" (production) | "text" (local dev)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Singleton settings — cached after first call."""
    return Settings()
