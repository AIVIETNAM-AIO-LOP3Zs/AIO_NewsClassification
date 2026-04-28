"""
Model factory — creates, caches, and manages the lifecycle of all
classifier backends.

Registry
--------
Supported model types:

    "tfidf" → TFIDFClassifier   (sklearn + joblib)
    "bert"  → BERTClassifier    (HuggingFace Transformers)

Adding a new backend
--------------------
1. Subclass ``BaseClassifier`` in a new ``app/ml/<name>_handler.py``.
2. Register it in ``_BACKEND_REGISTRY`` below.
3. Add the corresponding config keys to ``app/core/config.py``.

Lifecycle
---------
``ModelFactory`` is called once during the FastAPI lifespan::

    await ModelFactory.load_configured()   # startup
    await ModelFactory.unload_all()        # shutdown

During request handling, route handlers obtain a handler via::

    handler = ModelFactory.get(model_type)
"""
from __future__ import annotations

from typing import Dict, Optional

from app.core.config import get_settings
from app.core.logging import get_logger
from app.ml.base import BaseClassifier

logger = get_logger(__name__)
settings = get_settings()

# ---------------------------------------------------------------------------
# Registry — map model_type string → handler class
# ---------------------------------------------------------------------------
# Import lazily inside the factory methods to avoid importing heavy
# libraries (torch, transformers) at module-load time when not configured.

_SUPPORTED_TYPES = {"tfidf", "bert"}


def _build_handler(model_type: str) -> BaseClassifier:
    """Instantiate a fresh handler for the given model type."""
    if model_type == "tfidf":
        from app.ml.tfidf_handler import TFIDFClassifier
        return TFIDFClassifier()
    if model_type == "bert":
        from app.ml.bert_handler import BERTClassifier
        return BERTClassifier()
    raise ValueError(
        f"Unknown model type '{model_type}'. "
        f"Supported types: {sorted(_SUPPORTED_TYPES)}."
    )


# ---------------------------------------------------------------------------
# ModelFactory
# ---------------------------------------------------------------------------


class ModelFactory:
    """
    Registry that maps model-type strings to loaded handler instances.

    All methods are **class-level** so no instantiation is required.
    """

    _registry: Dict[str, BaseClassifier] = {}

    # ── Lifecycle ────────────────────────────────────────────────────────

    @classmethod
    async def load_configured(cls) -> None:
        """
        Load the model type(s) specified in ``settings.ACTIVE_MODELS``.

        ``ACTIVE_MODELS`` is a comma-separated list (e.g. ``"tfidf"`` or
        ``"tfidf,bert"``).  Each type is loaded exactly once.
        """
        for mtype in settings.active_model_list:
            await cls._load_one(mtype)

    @classmethod
    async def _load_one(cls, model_type: str) -> None:
        if model_type in cls._registry and cls._registry[model_type].is_loaded:
            logger.info("Handler already loaded", model_type=model_type)
            return
        handler = _build_handler(model_type)
        await handler.load()
        cls._registry[model_type] = handler
        logger.info("Handler registered", model_type=model_type)

    @classmethod
    async def unload_all(cls) -> None:
        """Release all loaded handlers."""
        for model_type, handler in list(cls._registry.items()):
            await handler.unload()
            logger.info("Handler unloaded", model_type=model_type)
        cls._registry.clear()

    # ── Retrieval ─────────────────────────────────────────────────────────

    @classmethod
    def get(cls, model_type: Optional[str] = None) -> BaseClassifier:
        """
        Return the handler for *model_type*.

        - If *model_type* is ``None``, falls back to the first entry in
          ``settings.ACTIVE_MODELS`` (i.e. the default model).
        - Raises ``KeyError`` with a clear message if the requested type
          is not loaded.
        """
        resolved = model_type or settings.active_model_list[0]
        if resolved not in cls._registry or not cls._registry[resolved].is_loaded:
            available = [k for k, v in cls._registry.items() if v.is_loaded]
            raise KeyError(
                f"Model '{resolved}' is not loaded. "
                f"Loaded models: {available}. "
                f"Add '{resolved}' to ACTIVE_MODELS in .env to enable it."
            )
        return cls._registry[resolved]

    @classmethod
    def loaded_types(cls) -> list[str]:
        """Return list of currently loaded model type strings."""
        return [k for k, v in cls._registry.items() if v.is_loaded]

    @classmethod
    def is_any_loaded(cls) -> bool:
        return any(v.is_loaded for v in cls._registry.values())
