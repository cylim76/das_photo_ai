from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache


@dataclass(frozen=True, slots=True)
class Settings:
    app_name: str = "DAS Photo AI"
    version: str = "0.4.1"
    api_prefix: str = "/api/v1"
    host: str = "127.0.0.1"
    port: int = 8800
    default_engine: str = "mock"
    log_level: str = "INFO"
    max_candidates: int = 10
    max_image_mb: int = 25
    preload_model: bool = False
    paddle_device: str = "gpu:0"
    paddle_lang: str | None = None
    paddle_ocr_version: str = "PP-OCRv5"
    paddle_detection_model: str | None = None
    paddle_recognition_model: str | None = None
    seal_multi_orientation: bool = True
    container_check_digit_fallback: bool = False
    check_digit_model: str = "en_PP-OCRv5_mobile_rec"
    check_digit_min_candidate_score: float = 0.50
    check_digit_min_single_score: float = 0.65

    @property
    def max_image_bytes(self) -> int:
        return self.max_image_mb * 1024 * 1024


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_optional(name: str) -> str | None:
    value = os.getenv(name, "").strip()
    return value or None


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings(
        host=os.getenv("DAS_AI_HOST", "127.0.0.1"),
        port=int(os.getenv("DAS_AI_PORT", "8800")),
        default_engine=os.getenv("DAS_AI_ENGINE", "mock").strip().lower(),
        log_level=os.getenv("DAS_AI_LOG_LEVEL", "INFO").strip().upper(),
        max_candidates=max(1, int(os.getenv("DAS_AI_MAX_CANDIDATES", "10"))),
        max_image_mb=max(1, int(os.getenv("DAS_AI_MAX_IMAGE_MB", "25"))),
        preload_model=_env_bool("DAS_AI_PRELOAD_MODEL"),
        paddle_device=os.getenv("DAS_AI_PADDLE_DEVICE", "gpu:0").strip(),
        paddle_lang=_env_optional("DAS_AI_PADDLE_LANG"),
        paddle_ocr_version=os.getenv("DAS_AI_PADDLE_OCR_VERSION", "PP-OCRv5").strip(),
        paddle_detection_model=_env_optional("DAS_AI_PADDLE_DET_MODEL"),
        paddle_recognition_model=_env_optional("DAS_AI_PADDLE_REC_MODEL"),
        seal_multi_orientation=_env_bool(
            "DAS_AI_SEAL_MULTI_ORIENTATION",
            default=True,
        ),
        container_check_digit_fallback=_env_bool(
            "DAS_AI_CONTAINER_CHECK_DIGIT_FALLBACK",
            default=False,
        ),
        check_digit_model=os.getenv(
            "DAS_AI_CHECK_DIGIT_MODEL",
            "en_PP-OCRv5_mobile_rec",
        ).strip(),
        check_digit_min_candidate_score=min(
            1.0,
            max(0.0, float(os.getenv("DAS_AI_CHECK_DIGIT_MIN_CANDIDATE_SCORE", "0.50"))),
        ),
        check_digit_min_single_score=min(
            1.0,
            max(0.0, float(os.getenv("DAS_AI_CHECK_DIGIT_MIN_SINGLE_SCORE", "0.65"))),
        ),
    )
