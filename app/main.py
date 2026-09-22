from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.api.router import router
from app.config import get_settings
from app.engines import create_engine
from app.engines.base import EngineError, EngineUnavailableError


@asynccontextmanager
async def lifespan(_: FastAPI):
    settings = get_settings()
    if settings.preload_model and settings.default_engine.startswith("paddle_"):
        create_engine(settings.default_engine, settings)
    yield


def create_app() -> FastAPI:
    settings = get_settings()
    logging.basicConfig(
        level=getattr(logging, settings.log_level, logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    application = FastAPI(
        title=settings.app_name,
        version=settings.version,
        description=(
            "OCR field extraction service for DAS Photo. Supports Mock/JSON development "
            "flows and PaddleOCR image inference on CPU or GPU."
        ),
        lifespan=lifespan,
    )

    @application.exception_handler(EngineUnavailableError)
    async def handle_engine_unavailable(
        _: Request, exc: EngineUnavailableError
    ) -> JSONResponse:
        return JSONResponse(
            status_code=503,
            content={"error": "engine_unavailable", "detail": str(exc)},
        )

    @application.exception_handler(EngineError)
    async def handle_engine_error(_: Request, exc: EngineError) -> JSONResponse:
        return JSONResponse(
            status_code=422,
            content={"error": "engine_input_error", "detail": str(exc)},
        )

    @application.get("/", include_in_schema=False)
    def root() -> dict[str, str]:
        return {
            "service": settings.app_name,
            "version": settings.version,
            "docs": "/docs",
            "health": f"{settings.api_prefix}/health",
        }

    application.include_router(router, prefix=settings.api_prefix)
    return application


app = create_app()
