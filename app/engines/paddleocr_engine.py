from __future__ import annotations

import os
import tempfile
from dataclasses import replace
from io import BytesIO
from pathlib import Path
from typing import Any

from PIL import Image

from app.config import Settings
from app.domain import EngineInput, OCRDocument, OCRItem
from app.engines.base import EngineError, EngineUnavailableError, OcrEngine
from app.engines.json_adapter import adapt_ocr_json
from app.engines.paddle_runtime import PADDLE_RUNTIME_LOCK


_ALLOWED_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
_ORIENTATIONS = ("original", "cw90", "ccw90")


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
            with PADDLE_RUNTIME_LOCK:
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

            with PADDLE_RUNTIME_LOCK:
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

    def recognize_orientations(
        self,
        input_data: EngineInput,
        orientations: tuple[str, ...] = _ORIENTATIONS,
    ) -> dict[str, OCRDocument]:
        """Run OCR on normalized image orientations and keep original coordinates.

        Rotated OCR is deliberately implemented in the engine rather than the seal
        extractor. This keeps OCR concerns separate from field-ranking rules and
        lets every returned candidate still point to the original uploaded image.
        """

        if not input_data.image_bytes:
            raise EngineError(f"{self.name} engine requires an uploaded image")
        invalid = [item for item in orientations if item not in _ORIENTATIONS]
        if invalid:
            raise EngineError(f"unsupported image orientation: {invalid[0]}")

        try:
            from PIL import Image, ImageOps
        except ImportError as exc:
            raise EngineUnavailableError(
                "Pillow is required for multi-orientation seal recognition"
            ) from exc

        try:
            with Image.open(BytesIO(input_data.image_bytes)) as opened:
                normalized = ImageOps.exif_transpose(opened).convert("RGB")
                original_width, original_height = normalized.size
                with tempfile.TemporaryDirectory(prefix="das-photo-ai-orientations-") as name:
                    temporary_root = Path(name)
                    documents: dict[str, OCRDocument] = {}
                    for orientation in orientations:
                        if orientation == "original":
                            variant = normalized
                        elif orientation == "cw90":
                            variant = normalized.transpose(Image.Transpose.ROTATE_270)
                        else:
                            variant = normalized.transpose(Image.Transpose.ROTATE_90)

                        image_path = temporary_root / f"{orientation}.png"
                        variant.save(image_path, format="PNG")
                        document = self._recognize_path(
                            image_path,
                            input_data.image_filename,
                            orientation,
                        )
                        documents[orientation] = _restore_original_coordinates(
                            document,
                            orientation,
                            original_width,
                            original_height,
                        )
                    return documents
        except (EngineError, EngineUnavailableError):
            raise
        except Exception as exc:
            raise EngineError(f"PaddleOCR orientation inference failed: {exc}") from exc

    def recognize_images(
        self,
        images: dict[str, Image.Image],
        image_filename: str | None = None,
    ) -> dict[str, OCRDocument]:
        """Recognize prepared in-memory variants with the already-loaded OCR model."""

        try:
            with tempfile.TemporaryDirectory(prefix="das-ai-ocr-variants-") as name:
                root = Path(name)
                documents: dict[str, OCRDocument] = {}
                for variant, image in images.items():
                    image_path = root / f"{variant}.png"
                    image.convert("RGB").save(image_path, format="PNG")
                    documents[variant] = self._recognize_path(
                        image_path,
                        image_filename,
                        variant,
                    )
                return documents
        except EngineError:
            raise
        except Exception as exc:
            raise EngineError(f"PaddleOCR variant inference failed: {exc}") from exc

    def _recognize_path(
        self,
        image_path: Path,
        image_filename: str | None,
        orientation: str,
    ) -> OCRDocument:
        with PADDLE_RUNTIME_LOCK:
            results = list(self._pipeline.predict(str(image_path)))
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
                "filename": image_filename,
                "ocr_version": self._settings.paddle_ocr_version,
                "orientation": orientation,
            }
        )
        return OCRDocument(
            items=document.items,
            source=document.source,
            metadata=metadata,
        )


def _restore_original_coordinates(
    document: OCRDocument,
    orientation: str,
    original_width: int,
    original_height: int,
) -> OCRDocument:
    if orientation == "original":
        return document

    def restore_point(x: float, y: float) -> tuple[float, float]:
        if orientation == "cw90":
            original_x, original_y = y, original_height - x
        else:
            original_x, original_y = original_width - y, x
        return (
            min(float(original_width), max(0.0, original_x)),
            min(float(original_height), max(0.0, original_y)),
        )

    restored_items: list[OCRItem] = []
    for item in document.items:
        polygon = tuple(restore_point(x, y) for x, y in item.polygon)
        if not polygon and item.box is not None:
            left, top, right, bottom = item.box
            polygon = tuple(
                restore_point(x, y)
                for x, y in (
                    (left, top),
                    (right, top),
                    (right, bottom),
                    (left, bottom),
                )
            )
        if polygon:
            xs = [point[0] for point in polygon]
            ys = [point[1] for point in polygon]
            box = (min(xs), min(ys), max(xs), max(ys))
        else:
            box = None
        restored_items.append(replace(item, polygon=polygon, box=box))

    metadata = dict(document.metadata)
    metadata["coordinates"] = "original_image"
    return OCRDocument(
        items=tuple(restored_items),
        source=document.source,
        metadata=metadata,
    )
