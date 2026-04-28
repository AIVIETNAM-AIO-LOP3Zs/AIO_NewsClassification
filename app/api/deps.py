"""
Dependency injection helpers — v2 (multi-model aware).
"""
from functools import lru_cache

from fastapi import Depends

from app.core.config import Settings, get_settings
from app.services.classification import ClassificationService


def get_app_settings() -> Settings:
    return get_settings()


@lru_cache(maxsize=1)
def _get_service() -> ClassificationService:
    return ClassificationService()


def get_classification_service(
    _settings: Settings = Depends(get_app_settings),
) -> ClassificationService:
    return _get_service()


# ── Optional API-Key Auth ─────────────────────────────────────────────────
# Uncomment and set API_KEY in .env to protect endpoints.
#
# from fastapi import HTTPException, status
# from fastapi.security import APIKeyHeader
#
# _api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)
#
# async def verify_api_key(
#     api_key: str | None = Depends(_api_key_header),
#     settings: Settings = Depends(get_app_settings),
# ) -> str:
#     if api_key != settings.API_KEY:
#         raise HTTPException(
#             status_code=status.HTTP_403_FORBIDDEN,
#             detail="Invalid or missing API key.",
#         )
#     return api_key
