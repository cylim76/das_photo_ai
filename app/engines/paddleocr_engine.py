from __future__ import annotations

import os
import tempfile
import threading
from pathlib import Path
from typing import Any

from app.config import Settings
from app.domain import EngineInput, OCRDocument
from app.engines.base import EngineError, EngineUnavailableError, OcrEngine
from app.engines.json_adapter import adapt_ocr_json


_ALLOWED_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def _image_suffix(filename: str | None) -> str:
    suffix = Path(filename or "upload.jpg").suffix.lower()
    return suffix if suffix in _ALLOWED_SUFFIXES else ".jpg"


def _result_payload(result: Any) -> dict[str, Any]:
    payload = getattr(result, "json", None)
    if callable(payload):
        payload = payload()
    if payload is None and isinstance(result, dict):
        payload = result
    if not isinstance(payload, dict):
        raise EngineError("PaddleOCR returned a result without a JSON dictionary")
    return payload


class PaddleOcrEngine(OcrEngine):
    description = "Runs PaddleOCR on an uploaded image and returns normalized OCR items."

    def __init__(self, name: str, settings: Settings) -> None:
        if name not in {"paddle_cpu", "paddle_gpu"}:
            raise ValueError(f"invalid PaddleOCR engine name: {name}")
        self.name = name
        self._settings = settings
        self._device = "cpu" if name == "paddle_cpu" else settings.paddle_device
        self._predict_lock = threading.Lock()
        self._pipeline = self._create_pipeline()

    @property
    def ready(self) -> bool:
        return self._pipeline is not None

    @property
    def device(self) -> str:
        return self._device

    def _create_pipeline(self) -> Any:
        try:
            from paddleocr import PaddleOCR
        except ImportError as exc:
            raise EngineUnavailableError(
                "PaddleOCR is not installed. Use the P3 GPU container or install "
                "requirements-paddle.txt in a compatible PaddlePaddle environment."
            ) from exc

        kwargs: dict[str, Any] = {
            "use_doc_orientation_classify": False,
            "use_doc_unwarping": False,
            "use_textline_orientation": False,
            "device": self._device,
            "ocr_version": self._settings.paddle_ocr_version,
        }
        if self._settings.paddle_lang:
            kwargs["lang"] = self._settings.paddle_lang
        if self._settings.paddle_detection_model:
            kwargs["text_detection_model_name"] = self._settings.paddle_detection_model
        if self._settings.paddle_recognition_model:
            kwargs["text_recognition_model_name"] = self._settings.paddle_recognition_model
        try:
            return PaddleOCR(**kwargs)
        except Exception as exc:
            raise EngineUnavailableError(
                f"PaddleOCR failed to initialize on {self._device}: {exc}"
            ) from exc

    def recognize(self, input_data: EngineInput) -> OCRDocument:
        if not input_data.image_bytes:
            raise EngineError(f"{self.name} engine requires an uploaded image")

        temporary_path: str | None = None
        try:
            with tempfile.NamedTemporaryFile(
                prefix="das-photo-ai-",
                suffix=_image_suffix(input_data.image_filename),
                delete=False,
            ) as temporary:
                temporary.write(input_data.image_bytes)
                temporary_path = temporary.name

            with self._predict_lock:
                results = list(self._pipeline.predict(temporary_path))
            if not results:
                raise EngineError("PaddleOCR returned no result for the uploaded image")
            document = adapt_ocr_json(
                _result_payload(results[0]),
                source=f"paddleocr:{self._device}",
            )
            metadata = dict(document.metadata)
            metadata.update(
                {
                    "engine": self.name,
                    "device": self._device,
                    "filename": input_data.image_filename,
                    "ocr_version": self._settings.paddle_ocr_version,
                }
            )
            return OCRDocument(
                items=document.items,
                source=document.source,
                metadata=metadata,
            )
        except EngineError:
            raise
        except Exception as exc:
            raise EngineError(f"PaddleOCR inference failed: {exc}") from exc
        finally:
            if temporary_path:
                try:
                    os.unlink(temporary_path)
                except FileNotFoundError:
                    pass
