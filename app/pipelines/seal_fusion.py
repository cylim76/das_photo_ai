from __future__ import annotations

import re
from dataclasses import replace

from app.pipelines.common import ExtractionCandidate, clamp_score


_STRUCTURED_SEAL = re.compile(r"^[A-Z]{1,5}\d{4,10}[A-Z]?$")
_ORIENTATION_ORDER = {"original": 0, "cw90": 1, "ccw90": 2}


def _format_score(value: str) -> float:
    if _STRUCTURED_SEAL.fullmatch(value) and 6 <= len(value) <= 12:
        return 1.0
    if value.isdigit() and 6 <= len(value) <= 10:
        return min(0.90, 0.74 + (len(value) - 6) * 0.04)
    if 6 <= len(value) <= 12 and any(char.isalpha() for char in value):
        return 0.72
    return 0.40


def _is_extension(longer: str, shorter: str) -> bool:
    difference = len(longer) - len(shorter)
    return (
        longer.isdigit()
        and shorter.isdigit()
        and len(longer) <= 10
        and len(shorter) >= 5
        and 1 <= difference <= 4
        and (longer.startswith(shorter) or longer.endswith(shorter))
    )


def fuse_seal_candidates(
    candidates_by_orientation: dict[str, list[ExtractionCandidate]],
    max_candidates: int = 10,
) -> list[ExtractionCandidate]:
    """Rank seal candidates from multiple image orientations without ground truth.

    The fusion gives evidence to an exact value repeated across orientations and
    to a longer value that is supported by a high-confidence partial reading.
    This handles split seals such as ``23`` + ``0418426`` while retaining all
    source OCR evidence for review.
    """

    all_candidates: list[ExtractionCandidate] = []
    for orientation, candidates in candidates_by_orientation.items():
        all_candidates.extend(
            replace(candidate, orientation=orientation) for candidate in candidates
        )
    if not all_candidates:
        return []

    by_value: dict[str, list[ExtractionCandidate]] = {}
    for candidate in all_candidates:
        by_value.setdefault(candidate.value, []).append(candidate)

    fused: list[ExtractionCandidate] = []
    for value, same_value in by_value.items():
        representative = max(
            same_value,
            key=lambda candidate: (
                candidate.extraction_score,
                candidate.ocr_confidence,
                -_ORIENTATION_ORDER.get(candidate.orientation, 99),
            ),
        )
        orientations = tuple(
            sorted(
                {candidate.orientation for candidate in same_value},
                key=lambda item: _ORIENTATION_ORDER.get(item, 99),
            )
        )
        reasons: list[str] = []
        score = (
            representative.extraction_score * 0.48
            + representative.ocr_confidence * 0.32
            + _format_score(value) * 0.20
        )

        if len(orientations) > 1:
            score += min(0.12, 0.07 * (len(orientations) - 1))
            reasons.append("cross_orientation_consensus")

        supported_by_partial = any(
            _is_extension(value, other.value)
            and representative.ocr_confidence >= other.ocr_confidence - 0.10
            for other in all_candidates
            if other.value != value
        )
        if supported_by_partial:
            score += 0.14
            reasons.append("supported_complete_extension")

        shadowed_by_longer = any(
            _is_extension(other.value, value)
            and other.ocr_confidence >= representative.ocr_confidence - 0.10
            for other in all_candidates
            if other.value != value
        )
        if shadowed_by_longer:
            score -= 0.06
            reasons.append("shorter_than_supported_candidate")

        fused.append(
            replace(
                representative,
                extraction_score=clamp_score(score),
                supporting_orientations=orientations,
                selection_reasons=tuple(reasons),
            )
        )

    return sorted(
        fused,
        key=lambda candidate: (
            candidate.extraction_score,
            candidate.ocr_confidence,
            len(candidate.value),
        ),
        reverse=True,
    )[:max_candidates]
