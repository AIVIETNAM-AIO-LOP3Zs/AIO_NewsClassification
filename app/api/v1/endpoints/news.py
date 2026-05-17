"""
News classification endpoints — v1.

Model selection
---------------
Every classification endpoint accepts an optional ``model_type`` query
parameter (``"tfidf"`` | ``"bert"``) that overrides the server-side default.
If not provided, the first entry in ``ACTIVE_MODELS`` is used.

Example::

    POST /api/v1/news/classify?model_type=bert
"""
from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status

from app.api.deps import get_classification_service
from app.core.config import get_settings
from app.core.logging import get_logger
from app.ml.factory import ModelFactory
from app.schemas.news import (
    ClassificationResult,
    HealthResponse,
    ModelInfoResponse,
    ModelStatus,
    NewsBatchRequest,
    NewsBatchResponse,
    NewsClassifyRequest,
    NewsClassifyResponse,
)
from app.services.classification import ClassificationService

router = APIRouter()
logger = get_logger(__name__)
settings = get_settings()


# ── Health & Meta ─────────────────────────────────────────────────────────


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Service health check",
    tags=["Monitoring"],
)
async def health_check() -> HealthResponse:
    """Returns overall service health and list of loaded model backends."""
    return HealthResponse(
        status="healthy",
        loaded_models=ModelFactory.loaded_types(),
        app_version=settings.APP_VERSION,
    )


@router.get(
    "/model/info",
    response_model=ModelInfoResponse,
    summary="Loaded model metadata",
    tags=["Monitoring"],
)
async def model_info() -> ModelInfoResponse:
    """
    Returns per-backend metadata: classes, version, and load status.
    Also includes the confidence threshold and supported BBC categories.
    """
    statuses = []
    for mtype in settings.active_model_list:
        try:
            h = ModelFactory.get(mtype)
            statuses.append(
                ModelStatus(
                    model_type=mtype,  # type: ignore[arg-type]
                    is_loaded=h.is_loaded,
                    version=h.version,
                    classes=h.classes,
                )
            )
        except KeyError:
            statuses.append(
                ModelStatus(
                    model_type=mtype,  # type: ignore[arg-type]
                    is_loaded=False,
                    version="N/A",
                    classes=[],
                )
            )
    return ModelInfoResponse(
        active_models=statuses,
        confidence_threshold=settings.CONFIDENCE_THRESHOLD,
        max_text_length=settings.MAX_TEXT_LENGTH,
    )


# ── Classification Endpoints ──────────────────────────────────────────────


@router.post(
    "/classify",
    response_model=NewsClassifyResponse,
    status_code=status.HTTP_200_OK,
    summary="Classify a single BBC news article",
    tags=["Classification"],
)
async def classify_news(
    request: NewsClassifyRequest,
    model_type: Optional[str] = Query(
        default=None,
        description="Model backend override: 'tfidf' | 'bert'",
        examples=["tfidf"],
    ),
    svc: ClassificationService = Depends(get_classification_service),
) -> NewsClassifyResponse:
    """
    Classify a single BBC news article into one of five categories:
    **sport**, **business**, **politics**, **tech**, **entertainment**.

    - **headline** (optional): article headline
    - **content** (required): article body text
    - **model_type** (optional): ``tfidf`` or ``bert`` — overrides server default
    """
    resolved_type = model_type or request.model_type
    try:
        result: ClassificationResult = svc.classify(request, model_type=resolved_type)
    except RuntimeError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        logger.exception("Unexpected error during classification", error=str(exc))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An internal error occurred during classification.",
        ) from exc

    logger.info(
        "Article classified",
        label=result.label,
        confidence=result.confidence,
        model_used=result.model_used,
        below_threshold=result.below_threshold,
    )
    return NewsClassifyResponse(
        result=result,
        model_version=svc.model_version(resolved_type),
    )


# ── Model-Specific Endpoints (Plan §2) ────────────────────────────────────


@router.post(
    "/classify/tfidf",
    response_model=NewsClassifyResponse,
    status_code=status.HTTP_200_OK,
    summary="Classify using TF-IDF SVC model",
    tags=["Classification"],
)
async def classify_tfidf(
    request: NewsClassifyRequest,
    svc: ClassificationService = Depends(get_classification_service),
) -> NewsClassifyResponse:
    """
    Classify a BBC news article using the **TF-IDF + LinearSVC** pipeline.

    This is a lightweight, CPU-optimised model ideal for low-latency inference.
    """
    try:
        result: ClassificationResult = svc.classify(request, model_type="tfidf")
    except RuntimeError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        logger.exception("Error in TF-IDF classification", error=str(exc))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An internal error occurred during TF-IDF classification.",
        ) from exc

    logger.info(
        "TF-IDF classification complete",
        label=result.label,
        confidence=result.confidence,
    )
    return NewsClassifyResponse(
        result=result,
        model_version=svc.model_version("tfidf"),
    )


@router.post(
    "/classify/bert",
    response_model=NewsClassifyResponse,
    status_code=status.HTTP_200_OK,
    summary="Classify using BERT model",
    tags=["Classification"],
)
async def classify_bert(
    request: NewsClassifyRequest,
    svc: ClassificationService = Depends(get_classification_service),
) -> NewsClassifyResponse:
    """
    Classify a BBC news article using the **BERT** transformer model.

    Provides superior accuracy with deep semantic understanding.
    Heavier on resources — benefits from GPU acceleration.
    """
    try:
        result: ClassificationResult = svc.classify(request, model_type="bert")
    except RuntimeError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        logger.exception("Error in BERT classification", error=str(exc))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An internal error occurred during BERT classification.",
        ) from exc

    logger.info(
        "BERT classification complete",
        label=result.label,
        confidence=result.confidence,
    )
    return NewsClassifyResponse(
        result=result,
        model_version=svc.model_version("bert"),
    )


# ── Batch (Legacy) ────────────────────────────────────────────────────────


@router.post(
    "/classify/batch",
    response_model=NewsBatchResponse,
    status_code=status.HTTP_200_OK,
    summary="Classify a batch of BBC news articles",
    tags=["Classification"],
)
async def classify_news_batch(
    request: NewsBatchRequest,
    model_type: Optional[str] = Query(
        default=None,
        description="Model backend override for the entire batch: 'tfidf' | 'bert'",
    ),
    svc: ClassificationService = Depends(get_classification_service),
) -> NewsBatchResponse:
    """
    Classify up to **32** BBC news articles in a single request.

    All articles in the batch are processed by the same model backend
    (query param → batch-level ``model_type`` field → server default).
    """
    resolved_type = model_type or request.model_type
    try:
        results = svc.classify_batch(request.articles, model_type=resolved_type)
    except RuntimeError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        logger.exception("Unexpected error during batch classification", error=str(exc))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An internal error occurred during batch classification.",
        ) from exc

    logger.info("Batch classified", total=len(results), model_type=resolved_type)
    return NewsBatchResponse(
        total=len(results),
        results=results,
        model_version=svc.model_version(resolved_type),
    )
