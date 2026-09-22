from __future__ import annotations

import re

from app.domain import OCRDocument
from app.pipelines.common import (
    ExtractionCandidate,
    clamp_score,
    compact_alphanumeric,
    deduplicate_candidates,
    generate_text_groups,
)


_DATE_PATTERN = re.compile(r"(?:19|20)\d{2}[./-]\d{1,2}[./-]\d{1,2}")
_TIME_PATTERN = re.compile(r"(?:[01]?\d|2[0-3]):[0-5]\d")
_CONTAINER_PREFIX_PATTERN = re.compile(r"^[A-Z]{3}[UJZ]\d{6,7}$")
_ISO_SIZE_PATTERN = re.compile(r"^\d{2}[A-Z]\d$")
_NOISE_SUBSTRINGS = {
    "MAXGROSS",
    "MAXPAYLOAD",
    "PAYLOAD",
    "TARE",
    "CUBE",
    "CUFT",
    "CUFT3",
    "KGS",
    "ACEP",
    "HAMBURG",
}
_NOISE_TOKENS = {
    "HMM",
    "COSCO",
    "OSCO",
    "SINOKOR",
    "EVERGREEN",
    "SHIPPING",
    "CMA",
    "CGM",
    "MSC",
    "ONE",
    "KG",
    "KGS",
    "LB",
    "LBS",
}


def _looks_like_noise(raw_text: str, compact: str, source_texts: tuple[str, ...]) -> bool:
    upper = raw_text.upper()
    if _DATE_PATTERN.search(upper) or _TIME_PATTERN.search(upper):
        return True
    if any(word in compact for word in _NOISE_SUBSTRINGS):
        return True
    normalized_sources = {compact_alphanumeric(text) for text in source_texts}
    if normalized_sources & _NOISE_TOKENS:
        return True
    if _CONTAINER_PREFIX_PATTERN.fullmatch(compact):
        return True
    if _ISO_SIZE_PATTERN.fullmatch(compact):
        return True
    if re.fullmatch(r"\d+(?:[.,]\d+)+(?:KG|KGS|LB|LBS)?", upper.replace(" ", "")):
        return True
    if len(compact) >= 8 and compact.isdigit() and compact.startswith(("19", "20")):
        return True
    return False


def extract_seal_numbers(
    document: OCRDocument,
    max_candidates: int = 10,
) -> list[ExtractionCandidate]:
    candidates: list[ExtractionCandidate] = []
    for group in generate_text_groups(document, max_items=2):
        compact = compact_alphanumeric(group.text)
        if not 5 <= len(compact) <= 20:
            continue
        if not any(character.isdigit() for character in compact):
            continue
        if _looks_like_noise(group.spaced_text, compact, group.source_texts):
            continue

        letters = sum(character.isalpha() for character in compact)
        digits = sum(character.isdigit() for character in compact)
        mixed = letters > 0 and digits > 0
        length_score = 1.0 if 6 <= len(compact) <= 12 else 0.65
        composition_score = 1.0 if mixed else 0.45
        # A complete seal is usually recognized as one OCR item. Combining items
        # remains available for split text, but must not outrank an equally good
        # single-item candidate merely because a nearby carrier logo adds letters.
        source_penalty = max(0, len(group.items) - 1) * 0.085
        all_digits_penalty = 0.10 if letters == 0 else 0.0
        very_long_penalty = 0.08 if len(compact) > 14 else 0.0

        extraction_score = clamp_score(
            group.confidence * 0.68
            + length_score * 0.14
            + composition_score * 0.18
            - source_penalty
            - all_digits_penalty
            - very_long_penalty
        )
        candidates.append(
            ExtractionCandidate(
                value=compact,
                observed_value=compact,
                ocr_confidence=round(group.confidence, 6),
                extraction_score=extraction_score,
                validation="heuristic",
                inferred=False,
                corrections=0,
                source_texts=group.source_texts,
                source_indices=group.source_indices,
                boxes=group.boxes,
            )
        )
    return deduplicate_candidates(candidates, max_candidates=max_candidates)
