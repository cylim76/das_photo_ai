from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


Point = tuple[float, float]
Polygon = tuple[Point, ...]
Box = tuple[float, float, float, float]


@dataclass(frozen=True, slots=True)
class OCRItem:
    text: str
    score: float
    polygon: Polygon = ()
    box: Box | None = None
    source_index: int = 0


@dataclass(frozen=True, slots=True)
class OCRDocument:
    items: tuple[OCRItem, ...]
    source: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class EngineInput:
    ocr_payload: dict[str, Any] | None = None
    image_bytes: bytes | None = None
    image_filename: str | None = None
    mock_profile: str = "default"

