"""
FastAPI application entry point.

Lifespan context
----------------
- startup : initialise logging → load all configured ML backends
- shutdown: release all ML artifacts
"""
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse

from app.api.v1.api import api_router
from app.core.config import get_settings
from app.core.logging import get_logger, setup_logging
from app.ml.factory import ModelFactory

settings = get_settings()

# Logging must be set up before the first log call
setup_logging(log_level=settings.LOG_LEVEL, log_format=settings.LOG_FORMAT)
logger = get_logger(__name__)


# ── Lifespan ──────────────────────────────────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Manage ML artifact lifecycle tied to the ASGI application lifespan."""
    # ── Startup ───────────────────────────────────────────────────────────
    logger.info(
        "Starting BBC News Classifier API",
        version=settings.APP_VERSION,
        active_models=settings.ACTIVE_MODELS,
        debug=settings.DEBUG,
    )
    await ModelFactory.load_configured()
    logger.info(
        "Startup complete",
        loaded_models=ModelFactory.loaded_types(),
    )

    yield  # ← application serves requests here

    # ── Shutdown ──────────────────────────────────────────────────────────
    logger.info("Shutting down — releasing ML artifacts.")
    await ModelFactory.unload_all()
    logger.info("Shutdown complete.")


# ── Application Factory ────────────────────────────────────────────────────


def create_app() -> FastAPI:
    app = FastAPI(
        title=settings.APP_NAME,
        version=settings.APP_VERSION,
        description=settings.APP_DESCRIPTION,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan,
    )

    # ── CORS ──────────────────────────────────────────────────────────────
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.ALLOWED_HOSTS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Routers ───────────────────────────────────────────────────────────
    app.include_router(api_router, prefix=settings.API_PREFIX)

    # ── Root → Swagger ────────────────────────────────────────────────────
    @app.get("/", include_in_schema=False)
    async def root() -> RedirectResponse:
        return RedirectResponse(url="/docs")

    return app


app = create_app()
