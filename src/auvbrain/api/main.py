"""API server entrypoint.

Run with:
    auv-api

Or directly:
    python -m auvbrain.api.main
"""

from __future__ import annotations

import sys

import uvicorn

from ..config import load_settings
from ..logging_config import configure_logging
from .app import create_app


def run() -> None:
    settings = load_settings()
    configure_logging(settings.log_level)

    app = create_app(settings)
    try:
        uvicorn.run(
            app,
            host=settings.api_host,
            port=settings.api_port,
            log_level=settings.log_level.lower(),
            # Use lifespan so startup/shutdown hooks fire correctly
            lifespan="on",
        )
    except KeyboardInterrupt:
        sys.exit(0)


if __name__ == "__main__":
    run()
