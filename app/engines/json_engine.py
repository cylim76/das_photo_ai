from __future__ import annotations

from app.domain import EngineInput, OCRDocument
from app.engines.base import EngineError, OcrEngine
from app.engines.json_adapter import adapt_ocr_json


class JsonEngine(OcrEngine):
    name = "json"
    description = "Reads an existing PaddleOCR JSON result; does not run OCR."

    def recognize(self, input_data: EngineInput) -> OCRDocument:
        if input_data.ocr_payload is None:
            raise EngineError("json engine requires ocr_result")
        return adapt_ocr_json(input_data.ocr_payload)

