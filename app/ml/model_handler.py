"""
Compatibility shim — ``ModelHandler`` now delegates to ``ModelFactory``.

.. deprecated::
    Import directly from ``app.ml.factory`` or ``app.ml.tfidf_handler`` /
    ``app.ml.bert_handler`` instead.  This module is retained only so that
    existing test fixtures referencing ``ModelHandler`` continue to work
    during the transition period.
"""
from app.ml.factory import ModelFactory  # noqa: F401

# Alias kept for backward compatibility with test fixtures
ModelHandler = ModelFactory
__all__ = ["ModelHandler", "ModelFactory"]
