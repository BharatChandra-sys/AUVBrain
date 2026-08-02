"""FastAPI application factory.

Lifespan
--------
On startup:
  - Initialise DB (create tables if not exist)
  - Configure rate limiter from settings
  - Attach settings to app.state for dependency injection

On shutdown:
  - Dispose DB engine cleanly
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from ..config import Settings, load_settings
from ..db.engine import close_db, init_db
from ..logging_config import configure_logging
from .limiter import configure_limiter
from .routes import router as api_router

logger = logging.getLogger(__name__)


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    settings: Settings = app.state.settings
    configure_logging(settings.log_level)
    logger.info("AUVBrain API starting up")

    await init_db(settings)
    configure_limiter(
        rate=float(settings.rate_limit_commands_per_s),
        burst=settings.rate_limit_commands_per_s * 2,
    )

    yield

    logger.info("AUVBrain API shutting down")
    await close_db()


def create_app(settings: Settings | None = None) -> FastAPI:
    if settings is None:
        settings = load_settings()

    app = FastAPI(
        title="AUVBrain Control Hub",
        version="0.2.0",
        description=(
            "Autonomous underwater vehicle decision engine + control hub. "
            "All write endpoints require a valid API key with write scope."
        ),
        lifespan=_lifespan,
    )

    app.state.settings = settings
    app.include_router(api_router)

    # ── Global exception handlers ─────────────────────────────────────────

    @app.exception_handler(Exception)
    async def _unhandled(request: Request, exc: Exception) -> JSONResponse:
        logger.exception("Unhandled exception on %s %s", request.method, request.url.path)
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal server error."},
        )

    return app
