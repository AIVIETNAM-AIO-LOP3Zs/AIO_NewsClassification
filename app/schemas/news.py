"""
Pydantic schemas for the BBC News Classification API.

BBC News categories (5 classes)
--------------------------------
sport | business | politics | tech | entertainment

Validation rules
----------------
- headline / content must not be blank after stripping
- content length ≤ settings.MAX_TEXT_LENGTH
- batch size ≤ 32 articles
- model_type must be one of the supported backend identifiers
"""
from __future__ import annotations

from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field, field_validator, model_validator

from app.core.config import get_settings

settings = get_settings()

# BBC News canonical category set (used for docs / examples)
BbcCategory = Literal["sport", "business", "politics", "tech", "entertainment"]

# Supported model backend identifiers
ModelType = Literal["tfidf", "bert"]

# ── Request Schemas ────────────────────────────────────────────────────────


class NewsClassifyRequest(BaseModel):
    """Single-article classification request."""

    headline: Optional[str] = Field(
        default=None,
        description="Article headline (optional; prepended to content when provided)",
        examples=["Arsenal win Premier League title after dramatic final day"],
    )
    content: str = Field(
        ...,
        min_length=1,
        description="Article body text",
        examples=[
            "Arsenal clinched the Premier League title on Sunday after a stunning "
            "comeback victory at the Emirates Stadium..."
        ],
    )
    model_type: Optional[ModelType] = Field(
        default=None,
        description=(
            "Model backend to use for this request. "
            "Overrides the server default (ACTIVE_MODELS[0]). "
            "Supported: 'tfidf', 'bert'."
        ),
        examples=["tfidf"],
    )

    @field_validator("content", "headline", mode="before")
    @classmethod
    def strip_whitespace(cls, v):
        return v.strip() if isinstance(v, str) else v

    @field_validator("content")
    @classmethod
    def validate_content_length(cls, v: str) -> str:
        max_len = settings.MAX_TEXT_LENGTH
        if len(v) > max_len:
            raise ValueError(
                f"Content exceeds the maximum length of {max_len} characters "
                f"(received {len(v)})."
            )
        return v

    @model_validator(mode="after")
    def check_not_blank(self) -> "NewsClassifyRequest":
        if not self.content:
            raise ValueError("'content' must not be blank.")
        return self

    def combined_text(self) -> str:
        """Merge headline + content for model input."""
        if self.headline:
            return f"{self.headline}. {self.content}"
        return self.content


class NewsBatchRequest(BaseModel):
    """Batch classification request (up to 32 articles per call)."""

    articles: List[NewsClassifyRequest] = Field(
        ...,
        min_length=1,
        max_length=32,
        description="List of articles to classify (maximum 32 per request)",
    )
    model_type: Optional[ModelType] = Field(
        default=None,
        description=(
            "Model backend for the entire batch. "
            "Per-article model_type fields are ignored when this is set."
        ),
    )


# ── Response Schemas ───────────────────────────────────────────────────────


class ClassificationResult(BaseModel):
    """Per-article classification output."""

    label: BbcCategory = Field(description="Predicted BBC News category")  # type: ignore[assignment]
    confidence: float = Field(
        ge=0.0, le=1.0,
        description="Confidence score of the top prediction (0–1)",
    )
    probabilities: Dict[str, float] = Field(
        description="Full per-class probability distribution"
    )
    below_threshold: bool = Field(
        default=False,
        description="True when confidence is below the configured threshold",
    )
    model_used: ModelType = Field(  # type: ignore[assignment]
        description="Model backend that produced this prediction"
    )


class NewsClassifyResponse(BaseModel):
    """Single-article response envelope."""

    status: str = "success"
    result: ClassificationResult
    model_version: str = Field(description="Artifact version / HuggingFace model ID")


class NewsBatchResponse(BaseModel):
    """Batch response envelope."""

    status: str = "success"
    total: int = Field(description="Number of articles processed")
    results: List[ClassificationResult]
    model_version: str


# ── Health / Meta Schemas ─────────────────────────────────────────────────


class ModelStatus(BaseModel):
    """Status of a single loaded model backend."""

    model_type: ModelType  # type: ignore[assignment]
    is_loaded: bool
    version: str
    classes: List[str]


class HealthResponse(BaseModel):
    status: str = "healthy"
    loaded_models: List[str]
    app_version: str


class ModelInfoResponse(BaseModel):
    active_models: List[ModelStatus]
    confidence_threshold: float
    max_text_length: int
    supported_categories: List[str] = list(BbcCategory.__args__)  # type: ignore[attr-defined]
