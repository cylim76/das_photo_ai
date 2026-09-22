from __future__ import annotations

import re
from dataclasses import dataclass

from app.domain import Box, OCRDocument, OCRItem


@dataclass(frozen=True, slots=True)
class TextGroup:
    items: tuple[OCRItem, ...]

    @property
    def text(self) -> str:
        return "".join(item.text for item in self.items)

    @property
    def spaced_text(self) -> str:
        return " ".join(item.text for item in self.items)

    @property
    def confidence(self) -> float:
        if not self.items:
            return 0.0
        return sum(item.score for item in self.items) / len(self.items)

    @property
    def source_indices(self) -> tuple[int, ...]:
        return tuple(item.source_index for item in self.items)

    @property
    def source_texts(self) -> tuple[str, ...]:
        return tuple(item.text for item in self.items)

    @property
    def boxes(self) -> tuple[Box, ...]:
        return tuple(item.box for item in self.items if item.box is not None)


@dataclass(frozen=True, slots=True)
class ExtractionCandidate:
    value: str
    observed_value: str
    ocr_confidence: float
    extraction_score: float
    validation: str | None
    inferred: bool
    corrections: int
    source_texts: tuple[str, ...]
    source_indices: tuple[int, ...]
    boxes: tuple[Box, ...]
    suggested_value: str | None = None
    verification: str = "not_applicable"
    observed_check_digit: str | None = None
    calculated_check_digit: str | None = None
    check_digit_source: str | None = None


def compact_alphanumeric(value: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", value.upper())


def _center(box: Box) -> tuple[float, float]:
    return (box[0] + box[2]) / 2, (box[1] + box[3]) / 2


def _dimensions(box: Box) -> tuple[float, float]:
    return max(1.0, box[2] - box[0]), max(1.0, box[3] - box[1])


def _spatially_related(left: OCRItem, right: OCRItem) -> bool:
    if left.box is None or right.box is None:
        return False
    left_x, left_y = _center(left.box)
    right_x, right_y = _center(right.box)
    left_w, left_h = _dimensions(left.box)
    right_w, right_h = _dimensions(right.box)

    same_row = abs(left_y - right_y) <= max(left_h, right_h) * 1.5
    row_gap = max(0.0, max(left.box[0], right.box[0]) - min(left.box[2], right.box[2]))
    if same_row and row_gap <= max(250.0, max(left_w, right_w) * 3.0):
        return True

    same_column = abs(left_x - right_x) <= max(left_w, right_w) * 1.5
    column_gap = max(0.0, max(left.box[1], right.box[1]) - min(left.box[3], right.box[3]))
    return same_column and column_gap <= max(250.0, max(left_h, right_h) * 4.0)


def generate_text_groups(document: OCRDocument, max_items: int = 3) -> list[TextGroup]:
    items = list(document.items)
    groups: list[TextGroup] = []
    seen: set[tuple[int, ...]] = set()

    def add(group_items: tuple[OCRItem, ...]) -> None:
        key = tuple(item.source_index for item in group_items)
        if key and key not in seen:
            seen.add(key)
            groups.append(TextGroup(group_items))

    for item in items:
        add((item,))

    for length in range(2, max_items + 1):
        for start in range(0, len(items) - length + 1):
            add(tuple(items[start : start + length]))

    positioned = [item for item in items if item.box is not None]
    by_reading_order = sorted(
        positioned,
        key=lambda item: (
            _center(item.box)[1] if item.box else 0,
            _center(item.box)[0] if item.box else 0,
        ),
    )
    for left_index, left in enumerate(by_reading_order):
        for right in by_reading_order[left_index + 1 :]:
            if _spatially_related(left, right):
                ordered = sorted(
                    (left, right),
                    key=lambda item: (
                        _center(item.box)[0] if item.box else 0,
                        _center(item.box)[1] if item.box else 0,
                    ),
                )
                add(tuple(ordered))

    return groups


def clamp_score(value: float) -> float:
    return round(min(1.0, max(0.0, value)), 6)


def deduplicate_candidates(
    candidates: list[ExtractionCandidate],
    max_candidates: int,
) -> list[ExtractionCandidate]:
    best_by_value: dict[str, ExtractionCandidate] = {}
    for candidate in candidates:
        current = best_by_value.get(candidate.value)
        if current is None or (
            candidate.extraction_score,
            candidate.ocr_confidence,
            -candidate.corrections,
        ) > (
            current.extraction_score,
            current.ocr_confidence,
            -current.corrections,
        ):
            best_by_value[candidate.value] = candidate
    return sorted(
        best_by_value.values(),
        key=lambda candidate: (
            candidate.extraction_score,
            candidate.ocr_confidence,
            -candidate.corrections,
        ),
        reverse=True,
    )[:max_candidates]
