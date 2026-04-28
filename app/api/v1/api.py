"""
v1 API router — aggregates all v1 endpoint routers.
"""
from fastapi import APIRouter

from app.api.v1.endpoints.news import router as news_router

api_router = APIRouter()

api_router.include_router(news_router, prefix="/news", tags=["Classification"])
