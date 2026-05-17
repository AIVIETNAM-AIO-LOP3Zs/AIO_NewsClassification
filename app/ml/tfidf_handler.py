"""
TF-IDF + scikit-learn classifier handler for BBC News classification.

Expected artifacts (joblib-serialised) under ``settings.TFIDF_MODEL_DIR``:
- ``TFIDF_VECTORIZER_NAME``   — fitted TfidfVectorizer
- ``TFIDF_MODEL_NAME``        — trained sklearn estimator
- ``TFIDF_LABEL_ENCODER_NAME``— LabelEncoder (optional)

Thread-safety: sklearn-fitted objects are read-only; concurrent calls are safe.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import joblib
import numpy as np

from app.core.config import get_settings
from app.core.logging import get_logger
from app.ml.base import BaseClassifier

logger = get_logger(__name__)
settings = get_settings()


class TFIDFClassifier(BaseClassifier):
    """
    Inference handler for a TF-IDF vectorised sklearn classifier.

    The handler is a plain object — singleton behaviour is managed
    by ``ModelFactory``.

    Usage::

        handler = TFIDFClassifier()
        await handler.load()
        result = handler.predict("Tottenham beat Arsenal 3-0 ...")
    """

    def __init__(self) -> None:
        self._vectorizer = None
        self._model = None
        self._label_encoder = None
        self._classes: List[str] = []
        self._loaded: bool = False

    # ── Lifecycle ────────────────────────────────────────────────────────

    async def load(self) -> None:
        if self._loaded:
            logger.info("TFIDFClassifier: already loaded, skipping.")
            return

        model_dir = Path(settings.TFIDF_MODEL_DIR)
        logger.info("Loading TF-IDF artifacts", dir=str(model_dir))

        self._vectorizer = self._load_artifact(
            model_dir / settings.TFIDF_VECTORIZER_NAME, required=True
        )
        self._model = self._load_artifact(
            model_dir / settings.TFIDF_MODEL_NAME, required=True
        )
        self._label_encoder = self._load_artifact(
            model_dir / settings.TFIDF_LABEL_ENCODER_NAME, required=False
        )

        # Resolve class labels
        if self._label_encoder is not None:
            self._classes = list(self._label_encoder.classes_)
        elif hasattr(self._model, "classes_"):
            self._classes = [str(c) for c in self._model.classes_]
        else:
            self._classes = []

        self._loaded = True
        logger.info(
            "TFIDFClassifier loaded",
            classes=self._classes,
            n_classes=len(self._classes),
        )

    async def unload(self) -> None:
        self._model = None
        self._vectorizer = None
        self._label_encoder = None
        self._loaded = False
        logger.info("TFIDFClassifier: artifacts released.")

    # ── Inference ────────────────────────────────────────────────────────

    def predict(self, text: str) -> Dict:
        """Vectorise ``text`` and return the top predicted class."""
        self._assert_loaded()

        X = self._vectorizer.transform([text])
        raw_pred = self._model.predict(X)[0]

        proba = self._get_proba(X)
        label_name = self._resolve_label(raw_pred)

        return {
            "label": label_name,
            "confidence": float(round(float(max(proba)), 6)),
            "probabilities": {
                cls: float(round(float(p), 6))
                for cls, p in zip(self._classes or [label_name], proba)
            },
        }

    def predict_batch(self, texts: List[str]) -> List[Dict]:
        """Vectorise all texts in one shot (more efficient than looping)."""
        self._assert_loaded()

        X = self._vectorizer.transform(texts)
        raw_preds = self._model.predict(X)

        if hasattr(self._model, "predict_proba"):
            probas = self._model.predict_proba(X)
        else:
            df = self._model.decision_function(X)
            df_arr = np.asarray(df)
            if df_arr.ndim == 1:
                p = 1 / (1 + np.exp(-df_arr))
                probas = np.column_stack([1 - p, p])
            else:
                e = np.exp(df_arr - df_arr.max(axis=1, keepdims=True))
                probas = e / e.sum(axis=1, keepdims=True)

        results = []
        for raw_pred, proba in zip(raw_preds, probas):
            label_name = self._resolve_label(raw_pred)
            results.append({
                "label": label_name,
                "confidence": float(round(float(max(proba)), 6)),
                "probabilities": {
                    cls: float(round(float(p), 6))
                    for cls, p in zip(self._classes or [label_name], proba)
                },
            })
        return results

    # ── Properties ───────────────────────────────────────────────────────

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def classes(self) -> List[str]:
        return self._classes

    @property
    def model_type(self) -> str:
        return "tfidf"

    @property
    def version(self) -> str:
        return settings.TFIDF_MODEL_NAME.removesuffix(".pkl")

    # ── Private ──────────────────────────────────────────────────────────

    def _get_proba(self, X) -> np.ndarray:
        if hasattr(self._model, "predict_proba"):
            return self._model.predict_proba(X)[0]
        # Multi-class LinearSVC: decision_function → softmax approximation
        df = np.asarray(self._model.decision_function(X))
        if df.ndim == 1:
            # Binary case
            p = 1 / (1 + np.exp(-df))
            return np.array([1 - p, p])
        # Multi-class: apply softmax for pseudo-probabilities
        df = df[0]  # shape: (n_classes,)
        e = np.exp(df - df.max())  # numerical stability
        return e / e.sum()

    def _resolve_label(self, raw_pred) -> str:
        if self._label_encoder is not None:
            return str(self._label_encoder.inverse_transform([raw_pred])[0])
        return str(raw_pred)

    @staticmethod
    def _load_artifact(path: Path, required: bool = True):
        if not path.exists():
            if required:
                raise FileNotFoundError(
                    f"Required TF-IDF artifact not found: {path}\n"
                    f"Train a model and save it to '{path.parent}/' using joblib.dump()."
                )
            logger.warning("Optional artifact not found", path=str(path))
            return None
        artifact = joblib.load(path)
        logger.info(
            "Artifact loaded",
            path=str(path),
            type=type(artifact).__name__,
        )
        return artifact
