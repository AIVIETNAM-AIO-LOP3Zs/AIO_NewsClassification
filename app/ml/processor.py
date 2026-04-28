"""
English text pre-processing pipeline for BBC News classification.

Pipeline steps (TF-IDF path)
-----------------------------
1. Unicode normalisation (NFC)
2. Lowercase
3. URL / email / HTML removal
4. Punctuation & digit stripping
5. Extra-whitespace collapse
6. English stopword removal (built-in set; swap for NLTK if preferred)
7. Whitespace tokenisation

NOTE: The BERT handler does **not** use this processor — it has its own
HuggingFace tokeniser that handles text normalisation internally.
"""
from __future__ import annotations

import re
import unicodedata
from typing import List

from app.core.logging import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# BBC News categories (reference)
# ---------------------------------------------------------------------------
BBC_CATEGORIES = frozenset({"sport", "business", "politics", "tech", "entertainment"})

# ---------------------------------------------------------------------------
# English stopwords (minimal built-in set)
# Swap the body of this set for ``nltk.corpus.stopwords.words("english")``
# if you want the full 179-word list without adding a dependency.
# ---------------------------------------------------------------------------
_ENGLISH_STOPWORDS: set[str] = {
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "are", "was", "were", "be", "been",
    "being", "have", "has", "had", "do", "does", "did", "will", "would",
    "could", "should", "may", "might", "shall", "can", "not", "no", "nor",
    "so", "yet", "both", "either", "neither", "each", "few", "more", "most",
    "other", "some", "such", "than", "too", "very", "just", "it", "its",
    "this", "that", "these", "those", "i", "we", "you", "he", "she", "they",
    "me", "us", "him", "her", "them", "my", "your", "his", "our", "their",
    "what", "which", "who", "whom", "when", "where", "why", "how", "all",
    "any", "both", "as", "if", "then", "because", "while", "although",
    "about", "above", "after", "before", "between", "into", "through",
    "up", "down", "out", "off", "over", "under", "again", "further",
    "s", "t", "re", "ve", "ll", "d",  # contraction remnants
}

# ---------------------------------------------------------------------------
# Compiled regex patterns
# ---------------------------------------------------------------------------
_RE_URL = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
_RE_EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[a-z]{2,}", re.IGNORECASE)
_RE_HTML = re.compile(r"<[^>]+>")
_RE_APOSTROPHE = re.compile(r"'s|'t|'re|'ve|'ll|'d|'m", re.IGNORECASE)
_RE_NON_ALPHA = re.compile(r"[^a-z\s]")
_RE_WHITESPACE = re.compile(r"\s+")


class TextProcessor:
    """
    Stateless text pre-processing utility for TF-IDF based classification.

    This class is **not** used by the BERT handler.

    Parameters
    ----------
    remove_stopwords : bool
        Whether to remove English stopwords during tokenisation.

    Usage
    -----
    processor = TextProcessor()
    clean    = processor.clean("BBC Sport: Tottenham beat Arsenal 3-0!")
    tokens   = processor.tokenize(clean)
    ready    = processor.process("Tottenham beat Arsenal 3-0 in the...")
    """

    def __init__(self, remove_stopwords: bool = True) -> None:
        self.remove_stopwords = remove_stopwords
        logger.info("TextProcessor initialised", remove_stopwords=remove_stopwords)

    # ── Public API ────────────────────────────────────────────────────────

    def clean(self, text: str) -> str:
        """Full cleaning pipeline → normalised lowercase string."""
        text = self._normalise(text)
        text = self._remove_urls(text)
        text = self._remove_html(text)
        text = self._expand_contractions(text)
        text = self._remove_non_alpha(text)
        text = self._collapse_whitespace(text)
        return text

    def tokenize(self, text: str) -> List[str]:
        """Split on whitespace; optionally drop stopwords."""
        tokens = text.split()
        if self.remove_stopwords:
            tokens = [t for t in tokens if t not in _ENGLISH_STOPWORDS and len(t) > 1]
        return tokens

    def process(self, text: str) -> str:
        """
        Full pipeline: ``clean`` → ``tokenize`` → rejoin.
        Returns a single string ready to be fed into a TfidfVectorizer.
        """
        return " ".join(self.tokenize(self.clean(text)))

    # ── Private helpers ───────────────────────────────────────────────────

    @staticmethod
    def _normalise(text: str) -> str:
        return unicodedata.normalize("NFC", text).lower()

    @staticmethod
    def _remove_urls(text: str) -> str:
        text = _RE_URL.sub(" ", text)
        return _RE_EMAIL.sub(" ", text)

    @staticmethod
    def _remove_html(text: str) -> str:
        return _RE_HTML.sub(" ", text)

    @staticmethod
    def _expand_contractions(text: str) -> str:
        """Strip common English contraction suffixes before removing non-alpha."""
        return _RE_APOSTROPHE.sub("", text)

    @staticmethod
    def _remove_non_alpha(text: str) -> str:
        return _RE_NON_ALPHA.sub(" ", text)

    @staticmethod
    def _collapse_whitespace(text: str) -> str:
        return _RE_WHITESPACE.sub(" ", text).strip()
