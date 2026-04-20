from __future__ import annotations

from fastapi import FastAPI

from .routes import router as api_router


def create_app() -> FastAPI:
    app = FastAPI(title="AUVBrain Control Hub", version="0.1.0")
    app.include_router(api_router)
    return app
