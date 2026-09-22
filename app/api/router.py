from __future__ import annotations

import re
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from starlette.concurrency import run_in_threadpool

from app.config import Settings, get_settings
from app.engines import available_engines
from app.engines.check_digit_recognizer import is_check_digit_recognizer_loaded
from app.schemas.recognition import (
    EngineName,
    RecognizeRequest,
    RecognizeResponse,
    TargetName,
)
from app.service import EXTRACTOR_VERSION, recognize, recognize_image


router = APIRouter()
_TARGETS: set[str] = {"container_number", "seal_number"}
_IMAGE_CONTENT_TYPES = {
    "image/jpeg",
    "image/png",
    "image/bmp",
    "image/webp",
    "image/tiff",
    "application/octet-stream",
}


def _parse_targets(value: str) -> list[TargetName]:
    parts = [part for part in re.split(r"[,;\s]+", value.strip()) if part]
    if not parts:
        raise HTTPException(status_code=422, detail="targets must not be empty")
    invalid = sorted(set(parts) - _TARGETS)
    if invalid:
        raise HTTPException(
            status_code=422,
            detail=f"unsupported targets: {', '.join(invalid)}",
        )
    return list(dict.fromkeys(parts))  # type: ignore[return-value]


@router.get("/health", tags=["system"])
def health(settings: Settings = Depends(get_settings)) -> dict[str, object]:
    return {
        "status": "ok",
        "service": settings.app_name,
        "version": settings.version,
        "extractor_version": EXTRACTOR_VERSION,
        "default_engine": settings.default_engine,
        "time": datetime.now(timezone.utc).isoformat(),
    }


@router.get("/models", tags=["system"])
def models(settings: Settings = Depends(get_settings)) -> dict[str, object]:
    engines = available_engines(settings)
    paddle_engines = [item for item in engines if str(item["name"]).startswith("paddle_")]
    check_digit_device = (
        "cpu" if settings.default_engine == "paddle_cpu" else settings.paddle_device
    )
    return {
        "default_engine": settings.default_engine,
        "engines": engines,
        "paddleocr": {
            "available": any(bool(item["available"]) for item in paddle_engines),
            "ocr_version": settings.paddle_ocr_version,
            "language": settings.paddle_lang,
            "gpu_device": settings.paddle_device,
            "seal_multi_orientation": settings.seal_multi_orientation,
        },
        "container_check_digit": {
            "enabled": settings.container_check_digit_fallback,
            "available": any(bool(item["available"]) for item in paddle_engines),
            "loaded": is_check_digit_recognizer_loaded(
                settings.check_digit_model,
                check_digit_device,
            ),
            "model_name": settings.check_digit_model,
            "device": check_digit_device,
            "minimum_candidate_score": settings.check_digit_min_candidate_score,
            "minimum_single_score": settings.check_digit_min_single_score,
        },
    }


@router.post("/recognize", response_model=RecognizeResponse, tags=["recognition"])
def recognize_endpoint(
    request: RecognizeRequest,
    settings: Settings = Depends(get_settings),
) -> RecognizeResponse:
    return recognize(request, settings)


@router.post("/recognize/image", response_model=RecognizeResponse, tags=["recognition"])
async def recognize_image_endpoint(
    image: UploadFile = File(..., description="JPG, PNG, BMP, WebP or TIFF image"),
    targets: str = Form("container_number,seal_number"),
    engine: EngineName | None = Form(None),
    request_id: str | None = Form(None, max_length=100),
    settings: Settings = Depends(get_settings),
) -> RecognizeResponse:
    content_type = (image.content_type or "application/octet-stream").lower()
    if content_type not in _IMAGE_CONTENT_TYPES:
        raise HTTPException(status_code=415, detail=f"unsupported image type: {content_type}")

    image_bytes = await image.read(settings.max_image_bytes + 1)
    await image.close()
    if not image_bytes:
        raise HTTPException(status_code=422, detail="uploaded image is empty")
    if len(image_bytes) > settings.max_image_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"image exceeds the {settings.max_image_mb} MB limit",
        )

    engine_name = engine or settings.default_engine
    if engine_name == "json":
        raise HTTPException(
            status_code=422,
            detail="json engine accepts OCR JSON at /api/v1/recognize, not an image",
        )
    parsed_targets = _parse_targets(targets)
    return await run_in_threadpool(
        recognize_image,
        image_bytes=image_bytes,
        image_filename=image.filename,
        targets=parsed_targets,
        engine_name=engine_name,
        request_id=request_id,
        settings=settings,
    )
