from __future__ import annotations

import importlib.util
import threading
from functools import lru_cache

from app.config import Settings, get_settings
from app.engines.base import EngineError, OcrEngine
from app.engines.check_digit_recognizer import clear_check_digit_recognizer_cache
from app.engines.json_engine import JsonEngine
from app.engines.mock_engine import MockEngine
from app.engines.paddleocr_engine import PaddleOcrEngine


_ENGINE_TYPES: dict[str, type[OcrEngine]] = {
    "mock": MockEngine,
    "json": JsonEngine,
}
_PADDLE_ENGINES = {"paddle_cpu", "paddle_gpu"}
_LOADED_ENGINES: set[str] = set()
_LOADED_LOCK = threading.Lock()


def is_paddle_available() -> bool:
    try:
        return importlib.util.find_spec("paddleocr") is not None
    except (ImportError, ValueError):
        return False


@lru_cache(maxsize=12)
def create_engine(name: str, settings: Settings | None = None) -> OcrEngine:
    normalized = name.strip().lower()
    resolved_settings = settings or get_settings()
    if normalized in _PADDLE_ENGINES:
        engine: OcrEngine = PaddleOcrEngine(normalized, resolved_settings)
    else:
        engine_type = _ENGINE_TYPES.get(normalized)
        if engine_type is None:
            supported = sorted(set(_ENGINE_TYPES) | _PADDLE_ENGINES)
            raise EngineError(
                f"unsupported engine '{name}'; available: {', '.join(supported)}"
            )
        engine = engine_type()
    with _LOADED_LOCK:
        _LOADED_ENGINES.add(normalized)
    return engine


def clear_engine_cache() -> None:
    create_engine.cache_clear()
    clear_check_digit_recognizer_cache()
    with _LOADED_LOCK:
        _LOADED_ENGINES.clear()


def available_engines(settings: Settings | None = None) -> list[dict[str, object]]:
    resolved_settings = settings or get_settings()
    with _LOADED_LOCK:
        loaded = set(_LOADED_ENGINES)
    engines: list[dict[str, object]] = []
    for name, engine_type in sorted(_ENGINE_TYPES.items()):
        engines.append(
            {
                "name": name,
                "available": True,
                "loaded": name in loaded,
                "device": None,
                "description": engine_type.description,
            }
        )
    paddle_available = is_paddle_available()
    for name in sorted(_PADDLE_ENGINES):
        device = "cpu" if name == "paddle_cpu" else resolved_settings.paddle_device
        engines.append(
            {
                "name": name,
                "available": paddle_available,
                "loaded": name in loaded,
                "device": device,
                "description": PaddleOcrEngine.description,
            }
        )
    return engines
