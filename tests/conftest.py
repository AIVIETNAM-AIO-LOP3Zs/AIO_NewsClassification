"""
Pytest fixtures shared across all tests.

Patching strategy
-----------------
Instead of touching the filesystem, we inject dummy TFIDFClassifier and
BERTClassifier stubs directly into ``ModelFactory._registry`` before each
test and clear the registry afterwards.
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.ml.base import BaseClassifier
from app.ml.factory import ModelFactory

# ---------------------------------------------------------------------------
# Dummy model stubs
# ---------------------------------------------------------------------------

BBC_CLASSES = ["sport", "business", "politics", "tech", "entertainment"]


class _DummyHandler(BaseClassifier):
    """Minimal stub that always predicts 'sport' with 90% confidence."""

    def __init__(self, mtype: str) -> None:
        self._mtype = mtype
        self._loaded = True

    async def load(self) -> None:
        self._loaded = True

    async def unload(self) -> None:
        self._loaded = False

    def predict(self, text: str) -> Dict:
        proba = [0.90, 0.025, 0.025, 0.025, 0.025]
        return {
            "label": "sport",
            "confidence": 0.90,
            "probabilities": dict(zip(BBC_CLASSES, proba)),
        }

    def predict_batch(self, texts: List[str]) -> List[Dict]:
        return [self.predict(t) for t in texts]

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def classes(self) -> List[str]:
        return BBC_CLASSES

    @property
    def model_type(self) -> str:
        return self._mtype

    @property
    def version(self) -> str:
        return f"dummy-{self._mtype}-v1"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def patch_model_factory():
    """
    Inject dummy handlers for both 'tfidf' and 'bert' before each test.
    Clears the registry after the test to avoid cross-test pollution.
    """
    ModelFactory._registry["tfidf"] = _DummyHandler("tfidf")
    ModelFactory._registry["bert"] = _DummyHandler("bert")
    yield
    ModelFactory._registry.clear()


@pytest.fixture
def client():
    """Synchronous TestClient with server exceptions propagated."""
    from app.main import create_app
    app = create_app()
    with TestClient(app, raise_server_exceptions=True) as c:
        yield c
