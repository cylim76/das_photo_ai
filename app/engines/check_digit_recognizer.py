from __future__ import annotations

import tempfile
import threading
import time
from functools import lru_cache
from pathlib import Path
from typing import Any

from PIL import Image

from app.engines.base import EngineError, EngineUnavailableError
from app.engines.paddle_runtime import PADDLE_RUNTIME_LOCK


def _result_payload(result: Any) -> dict[str, Any]:
    payload = getattr(result, "json", None)
    if callable(payload):
        payload = payload()
    if payload is None and isinstance(result, dict):
        payload = result
    if not isinstance(payload, dict):
        raise EngineError(
            "PaddleOCR TextRecognition returned a result without a JSON dictionary"
        )
    nested = payload.get("res")
    return nested if isinstance(nested, dict) else payload


class PaddleCheckDigitRecognizer:
    description = (
        "Runs a recognition-only English PaddleOCR model on localized check-digit images."
    )

    def __init__(self, model_name: str, device: str) -> None:
        self.model_name = model_name
        self.device = device
        try:
            from paddleocr import TextRecognition
        except ImportError as exc:
            raise EngineUnavailableError(
                "PaddleOCR TextRecognition is not installed. Use the P3 Paddle image."
            ) from exc
        try:
            with PADDLE_RUNTIME_LOCK:
                self._model = TextRecognition(model_name=model_name, device=device)
        except Exception as exc:
            raise EngineUnavailableError(
                f"check-digit model failed to initialize on {device}: {exc}"
            ) from exc

    def recognize(self, images: dict[str, Image.Image]) -> dict[str, dict[str, Any]]:
        if not images:
            return {}
        results: dict[str, dict[str, Any]] = {}
        try:
            with tempfile.TemporaryDirectory(prefix="das-ai-check-digit-") as name:
                root = Path(name)
                paths: dict[str, Path] = {}
                for variant, image in images.items():
                    path = root / f"{variant}.png"
                    image.convert("RGB").save(path, format="PNG")
                    paths[variant] = path

                with PADDLE_RUNTIME_LOCK:
                    for variant, path in paths.items():
                        started = time.perf_counter()
                        predictions = list(
                            self._model.predict(str(path), batch_size=1)
                        )
                        if not predictions:
                            raise EngineError(
                                f"TextRecognition returned no result for {variant}"
                            )
                        payload = _result_payload(predictions[0])
                        results[variant] = {
                            "text": str(payload.get("rec_text") or ""),
                            "score": round(float(payload.get("rec_score") or 0.0), 6),
                            "elapsed_ms": round(
                                (time.perf_counter() - started) * 1000,
                                3,
                            ),
                        }
            return results
        except (EngineError, EngineUnavailableError):
            raise
        except Exception as exc:
            raise EngineError(f"check-digit recognition failed: {exc}") from exc


_LOADED_KEYS: set[tuple[str, str]] = set()
_LOADED_LOCK = threading.Lock()


@lru_cache(maxsize=8)
def create_check_digit_recognizer(
    model_name: str,
    device: str,
) -> PaddleCheckDigitRecognizer:
    recognizer = PaddleCheckDigitRecognizer(model_name, device)
    with _LOADED_LOCK:
        _LOADED_KEYS.add((model_name, device))
    return recognizer


def is_check_digit_recognizer_loaded(model_name: str, device: str) -> bool:
    with _LOADED_LOCK:
        return (model_name, device) in _LOADED_KEYS


def clear_check_digit_recognizer_cache() -> None:
    create_check_digit_recognizer.cache_clear()
    with _LOADED_LOCK:
        _LOADED_KEYS.clear()
