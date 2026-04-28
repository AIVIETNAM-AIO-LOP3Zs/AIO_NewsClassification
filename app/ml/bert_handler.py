"""
BERT-based classifier handler for BBC News classification.

Supports both:
- A **local** fine-tuned model checkpoint (preferred for production)
- A raw **HuggingFace Hub** model ID (e.g. ``"bert-base-uncased"``) —
  useful for quick experimentation; requires internet access on first run.

Configuration (via .env)
------------------------
BERT_MODEL_DIR   — root directory for local checkpoints
BERT_MODEL_NAME  — local sub-dir name OR HuggingFace model id
BERT_MAX_LENGTH  — max tokeniser sequence length   (default: 512)
BERT_DEVICE      — "cpu" | "cuda" | "mps"          (default: "cpu")

Dependencies
------------
pip install torch transformers sentencepiece
(listed in requirements.txt under the [bert] extras section)

NOTE: If transformers is not installed, importing this module raises
``ImportError`` with a clear install message.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

from app.core.config import get_settings
from app.core.logging import get_logger
from app.ml.base import BaseClassifier

logger = get_logger(__name__)
settings = get_settings()

# ---------------------------------------------------------------------------
# Soft dependency check
# ---------------------------------------------------------------------------
try:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    _TRANSFORMERS_AVAILABLE = True
except ImportError:
    _TRANSFORMERS_AVAILABLE = False


def _require_transformers() -> None:
    if not _TRANSFORMERS_AVAILABLE:
        raise ImportError(
            "The 'transformers' and 'torch' packages are required for the BERT handler.\n"
            "Install them with:\n"
            "    pip install torch transformers sentencepiece\n"
            "or run: pip install -r requirements-bert.txt"
        )


# ---------------------------------------------------------------------------
# BERT Classifier Handler
# ---------------------------------------------------------------------------


class BERTClassifier(BaseClassifier):
    """
    Inference handler for a fine-tuned HuggingFace sequence-classification model.

    The handler is a plain object — singleton behaviour is managed by
    ``ModelFactory``.

    Inference pipeline
    ------------------
    1. Tokenise with ``AutoTokenizer``
    2. Forward pass through ``AutoModelForSequenceClassification``
    3. Softmax over logits → per-class probabilities
    4. argmax → label name via ``model.config.id2label``

    Usage::

        handler = BERTClassifier()
        await handler.load()
        result = handler.predict("Tottenham beat Arsenal 3-0 in the derby.")
    """

    def __init__(self) -> None:
        self._tokenizer = None
        self._model = None
        self._device: Optional[str] = None
        self._classes: List[str] = []
        self._loaded: bool = False

    # ── Lifecycle ────────────────────────────────────────────────────────

    async def load(self) -> None:
        _require_transformers()

        if self._loaded:
            logger.info("BERTClassifier: already loaded, skipping.")
            return

        model_id = self._resolve_model_id()
        self._device = self._resolve_device()

        logger.info(
            "Loading BERT model",
            model_id=model_id,
            device=self._device,
            max_length=settings.BERT_MAX_LENGTH,
        )

        self._tokenizer = AutoTokenizer.from_pretrained(model_id)
        self._model = AutoModelForSequenceClassification.from_pretrained(model_id)
        self._model.to(self._device)
        self._model.eval()  # disable dropout, etc.

        # Class labels from model config (set during fine-tuning)
        id2label: dict = getattr(self._model.config, "id2label", {})
        if id2label:
            self._classes = [id2label[i] for i in range(len(id2label))]
        else:
            logger.warning(
                "BERT model has no id2label config — "
                "class names will be numeric indices."
            )
            num_labels = self._model.config.num_labels
            self._classes = [str(i) for i in range(num_labels)]

        self._loaded = True
        logger.info(
            "BERTClassifier loaded",
            classes=self._classes,
            n_classes=len(self._classes),
            device=self._device,
        )

    async def unload(self) -> None:
        self._model = None
        self._tokenizer = None
        self._loaded = False
        logger.info("BERTClassifier: artifacts released.")

    # ── Inference ────────────────────────────────────────────────────────

    def predict(self, text: str) -> Dict:
        """Single-text inference."""
        self._assert_loaded()
        return self.predict_batch([text])[0]

    def predict_batch(self, texts: List[str]) -> List[Dict]:
        """
        Batch inference — tokenises all texts together for GPU efficiency.
        Falls back to CPU if CUDA is unavailable.
        """
        self._assert_loaded()
        _require_transformers()

        encoding = self._tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=settings.BERT_MAX_LENGTH,
            return_tensors="pt",
        )
        # Move tensors to the model device
        encoding = {k: v.to(self._device) for k, v in encoding.items()}

        with torch.no_grad():
            logits = self._model(**encoding).logits  # shape: (batch, num_labels)

        probs = torch.softmax(logits, dim=-1).cpu().numpy()  # (batch, num_labels)
        preds = probs.argmax(axis=-1)                        # (batch,)

        results = []
        for pred_idx, row_probs in zip(preds, probs):
            label_name = self._classes[int(pred_idx)]
            results.append({
                "label": label_name,
                "confidence": float(round(float(row_probs[pred_idx]), 6)),
                "probabilities": {
                    cls: float(round(float(p), 6))
                    for cls, p in zip(self._classes, row_probs)
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
        return "bert"

    @property
    def version(self) -> str:
        return settings.BERT_MODEL_NAME

    # ── Private ──────────────────────────────────────────────────────────

    def _resolve_model_id(self) -> str:
        """
        Return either a local filesystem path (if it exists) or the raw
        HuggingFace Hub model ID.
        """
        local_path = Path(settings.BERT_MODEL_DIR) / settings.BERT_MODEL_NAME
        if local_path.exists():
            logger.info("Using local BERT checkpoint", path=str(local_path))
            return str(local_path)
        # Fall back to HuggingFace Hub
        logger.info(
            "Local checkpoint not found — using HuggingFace Hub",
            model_id=settings.BERT_MODEL_NAME,
        )
        return settings.BERT_MODEL_NAME

    @staticmethod
    def _resolve_device() -> str:
        configured = settings.BERT_DEVICE.lower()
        if configured == "cuda":
            return "cuda" if torch.cuda.is_available() else "cpu"
        if configured == "mps":
            return "mps" if torch.backends.mps.is_available() else "cpu"  # type: ignore[attr-defined]
        return "cpu"
