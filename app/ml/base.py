"""
Abstract base class for all News Classification model handlers.

Any new model backend (TF-IDF, BERT, LLM, etc.) must subclass
``BaseClassifier`` and implement every abstract method.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List


class BaseClassifier(ABC):
    """
    Common interface for model backends.

    Lifecycle
    ---------
    await handler.load()       # called once at startup
    result = handler.predict() # called per request
    await handler.unload()     # called at shutdown
    """

    # ── Lifecycle ────────────────────────────────────────────────────────

    @abstractmethod
    async def load(self) -> None:
        """Load model artifacts from disk / HuggingFace Hub into memory."""

    @abstractmethod
    async def unload(self) -> None:
        """Release all resources held by this handler."""

    # ── Inference ────────────────────────────────────────────────────────

    @abstractmethod
    def predict(self, text: str) -> Dict:
        """
        Run inference on a **pre-processed** text string.

        Returns
        -------
        dict
            label        : str   — predicted class name
            confidence   : float — max probability (0-1)
            probabilities: dict[str, float] — full distribution
        """

    def predict_batch(self, texts: List[str]) -> List[Dict]:
        """
        Batch inference.  Default: loop over ``predict()``.
        Override in subclasses that support native batching (e.g. BERT).
        """
        return [self.predict(t) for t in texts]

    # ── Properties ───────────────────────────────────────────────────────

    @property
    @abstractmethod
    def is_loaded(self) -> bool:
        """True once artifacts are loaded and ready for inference."""

    @property
    @abstractmethod
    def classes(self) -> List[str]:
        """Ordered list of class label names."""

    @property
    @abstractmethod
    def model_type(self) -> str:
        """Short identifier: ``"tfidf"`` | ``"bert"``."""

    @property
    @abstractmethod
    def version(self) -> str:
        """Human-readable version / identifier for this artifact."""

    # ── Shared helper ────────────────────────────────────────────────────

    def _assert_loaded(self) -> None:
        if not self.is_loaded:
            raise RuntimeError(
                f"[{self.model_type}] Model is not loaded. "
                "Ensure the startup lifespan handler completed successfully."
            )
