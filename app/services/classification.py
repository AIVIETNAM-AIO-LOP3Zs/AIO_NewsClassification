"""
Classification business logic layer.

Orchestrates:  TextProcessor (TF-IDF path) → ModelFactory → threshold check
"""
from __future__ import annotations

from typing import List, Optional

from app.core.config import get_settings
from app.core.logging import get_logger
from app.ml.base import BaseClassifier
from app.ml.factory import ModelFactory
from app.ml.processor import TextProcessor
from app.schemas.news import ClassificationResult, NewsClassifyRequest

logger = get_logger(__name__)
settings = get_settings()

# TF-IDF text processor — instantiated once per worker process.
# BERT handlers use their own HuggingFace tokeniser internally.
_tfidf_processor = TextProcessor(remove_stopwords=True)


class ClassificationService:
    """
    Orchestrates the end-to-end classification pipeline.

    1. Resolve which model backend to use (per-request override → server default)
    2. Pre-process text (TF-IDF path uses TextProcessor; BERT path is raw)
    3. Invoke model inference
    4. Apply confidence threshold check
    5. Assemble typed response objects
    """

    def __init__(self) -> None:
        self._threshold: float = settings.CONFIDENCE_THRESHOLD

    # ── Public API ────────────────────────────────────────────────────────

    def classify(
        self,
        request: NewsClassifyRequest,
        model_type: Optional[str] = None,
    ) -> ClassificationResult:
        """Classify a single article."""
        handler = self._get_handler(model_type or request.model_type)
        text = self._prepare_text(request, handler)

        logger.debug(
            "Running inference",
            model_type=handler.model_type,
            text_len=len(text),
        )

        prediction = handler.predict(text)
        return self._build_result(prediction, handler.model_type)

    def classify_batch(
        self,
        requests: List[NewsClassifyRequest],
        model_type: Optional[str] = None,
    ) -> List[ClassificationResult]:
        """Classify a batch of articles using the same backend."""
        # Use a single handler for the whole batch (efficient for BERT GPU batching)
        first_model_type = model_type or (requests[0].model_type if requests else None)
        handler = self._get_handler(first_model_type)

        texts = [self._prepare_text(r, handler) for r in requests]
        predictions = handler.predict_batch(texts)

        logger.info(
            "Batch inference complete",
            model_type=handler.model_type,
            batch_size=len(predictions),
        )
        return [self._build_result(p, handler.model_type) for p in predictions]

    def model_version(self, model_type: Optional[str] = None) -> str:
        """Return the version string for the requested (or default) handler."""
        handler = self._get_handler(model_type)
        return handler.version

    # ── Private ──────────────────────────────────────────────────────────

    @staticmethod
    def _get_handler(model_type: Optional[str]) -> BaseClassifier:
        try:
            return ModelFactory.get(model_type)
        except KeyError as exc:
            raise RuntimeError(str(exc)) from exc

    @staticmethod
    def _prepare_text(request: NewsClassifyRequest, handler: BaseClassifier) -> str:
        """
        Return a text string ready for the given handler.

        - TF-IDF: clean & normalise via TextProcessor
        - BERT:   pass raw combined text (tokeniser handles normalisation)
        """
        raw = request.combined_text()
        if handler.model_type == "tfidf":
            return _tfidf_processor.process(raw)
        # BERT tokeniser handles its own normalisation
        return raw

    def _build_result(self, prediction: dict, model_type: str) -> ClassificationResult:
        confidence = prediction["confidence"]
        return ClassificationResult(
            label=prediction["label"],  # type: ignore[arg-type]
            confidence=confidence,
            probabilities=prediction["probabilities"],
            below_threshold=confidence < self._threshold,
            model_used=model_type,  # type: ignore[arg-type]
        )
