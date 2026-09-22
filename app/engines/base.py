from __future__ import annotations

from abc import ABC, abstractmethod

from app.domain import EngineInput, OCRDocument


class EngineError(ValueError):
    """Raised when an OCR engine cannot process its input."""


class EngineUnavailableError(EngineError):
    """Raised when an optional OCR runtime is not installed or cannot start."""


class OcrEngine(ABC):
    name: str
    description: str

    @property
    def ready(self) -> bool:
        return True

    @abstractmethod
    def recognize(self, input_data: EngineInput) -> OCRDocument:
        raise NotImplementedError
