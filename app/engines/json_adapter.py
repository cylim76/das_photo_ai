from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from app.domain import Box, OCRDocument, OCRItem, Polygon
from app.engines.base import EngineError


def _as_float(value: Any, field_name: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise EngineError(f"{field_name} contains a non-numeric value") from exc


def _polygon(value: Any) -> Polygon:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return ()
    points: list[tuple[float, float]] = []
    for point in value:
        if not isinstance(point, Sequence) or len(point) < 2:
            return ()
        points.append((_as_float(point[0], "polygon"), _as_float(point[1], "polygon")))
    return tuple(points)


def _box(value: Any) -> Box | None:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) < 4:
        return None
    return tuple(_as_float(value[i], "box") for i in range(4))  # type: ignore[return-value]


def _box_from_polygon(polygon: Polygon) -> Box | None:
    if not polygon:
        return None
    xs = [point[0] for point in polygon]
    ys = [point[1] for point in polygon]
    return min(xs), min(ys), max(xs), max(ys)


def _unwrap_payload(payload: dict[str, Any]) -> dict[str, Any]:
    if "rec_texts" in payload or "items" in payload:
        return payload
    for key in ("res", "result", "ocr_result", "data"):
        nested = payload.get(key)
        if isinstance(nested, dict) and ("rec_texts" in nested or "items" in nested):
            return nested
    raise EngineError("OCR JSON must contain rec_texts or items")


def _normalized_items(payload: dict[str, Any]) -> tuple[OCRItem, ...]:
    raw_items = payload.get("items")
    if not isinstance(raw_items, list):
        raise EngineError("items must be a list")

    items: list[OCRItem] = []
    for index, raw in enumerate(raw_items):
        if not isinstance(raw, dict):
            raise EngineError("each items entry must be an object")
        text = str(raw.get("text", "")).strip()
        if not text:
            continue
        polygon = _polygon(raw.get("polygon", []))
        box = _box(raw.get("box")) or _box_from_polygon(polygon)
        score = min(1.0, max(0.0, _as_float(raw.get("score", 0.0), "score")))
        items.append(
            OCRItem(
                text=text,
                score=score,
                polygon=polygon,
                box=box,
                source_index=int(raw.get("source_index", index)),
            )
        )
    return tuple(items)


def _paddle_items(payload: dict[str, Any]) -> tuple[OCRItem, ...]:
    texts = payload.get("rec_texts")
    scores = payload.get("rec_scores")
    polygons = payload.get("rec_polys", payload.get("dt_polys", []))
    boxes = payload.get("rec_boxes", [])

    if not isinstance(texts, list):
        raise EngineError("rec_texts must be a list")
    if scores is None:
        scores = [0.0] * len(texts)
    if not isinstance(scores, list) or len(scores) != len(texts):
        raise EngineError("rec_scores must be a list with the same length as rec_texts")
    if not isinstance(polygons, list):
        polygons = []
    if not isinstance(boxes, list):
        boxes = []

    items: list[OCRItem] = []
    for index, raw_text in enumerate(texts):
        text = str(raw_text).strip()
        if not text:
            continue
        polygon = _polygon(polygons[index]) if index < len(polygons) else ()
        box = _box(boxes[index]) if index < len(boxes) else _box_from_polygon(polygon)
        score = min(1.0, max(0.0, _as_float(scores[index], "rec_scores")))
        items.append(
            OCRItem(
                text=text,
                score=score,
                polygon=polygon,
                box=box,
                source_index=index,
            )
        )
    return tuple(items)


def adapt_ocr_json(payload: dict[str, Any], source: str = "paddleocr_json") -> OCRDocument:
    if not isinstance(payload, dict):
        raise EngineError("OCR payload must be a JSON object")
    unwrapped = _unwrap_payload(payload)
    items = (
        _normalized_items(unwrapped)
        if "items" in unwrapped
        else _paddle_items(unwrapped)
    )
    return OCRDocument(
        items=items,
        source=source,
        metadata={
            "input_path": unwrapped.get("input_path"),
            "text_type": unwrapped.get("text_type"),
        },
    )
