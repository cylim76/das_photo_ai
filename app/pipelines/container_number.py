from __future__ import annotations

from app.domain import OCRDocument
from app.pipelines.common import (
    ExtractionCandidate,
    clamp_score,
    compact_alphanumeric,
    deduplicate_candidates,
    generate_text_groups,
)
from app.validators.iso6346 import append_check_digit, validate_container_number


_DIGIT_TO_LETTER = {
    "0": "O",
    "1": "I",
    "2": "Z",
    "5": "S",
    "6": "G",
    "8": "B",
}
_LETTER_TO_DIGIT = {
    "O": "0",
    "Q": "0",
    "D": "0",
    "I": "1",
    "L": "1",
    "Z": "2",
    "S": "5",
    "G": "6",
    "B": "8",
}
_CATEGORY_CORRECTIONS = {"V": "U", "Y": "U"}


def _coerce_candidate(raw: str) -> tuple[str, int] | None:
    if len(raw) not in (10, 11):
        return None
    output: list[str] = []
    corrections = 0
    for index, character in enumerate(raw):
        if index < 3:
            if character.isalpha():
                output.append(character)
            elif character in _DIGIT_TO_LETTER:
                output.append(_DIGIT_TO_LETTER[character])
                corrections += 1
            else:
                return None
        elif index == 3:
            if character in "UJZ":
                output.append(character)
            elif character in _CATEGORY_CORRECTIONS:
                output.append(_CATEGORY_CORRECTIONS[character])
                corrections += 1
            else:
                return None
        else:
            if character.isdigit():
                output.append(character)
            elif character in _LETTER_TO_DIGIT:
                output.append(_LETTER_TO_DIGIT[character])
                corrections += 1
            else:
                return None
    return "".join(output), corrections


def _windows(value: str) -> list[str]:
    # An 11-character OCR token is an independently observed full number. Do
    # not also reinterpret its first 10 characters as a separate inferred
    # number, because that would hide an observed check-digit mismatch.
    if len(value) == 11:
        return [value]
    windows: list[str] = []
    for size in (11, 10):
        if len(value) < size:
            continue
        windows.extend(value[start : start + size] for start in range(len(value) - size + 1))
    return windows


def extract_container_numbers(
    document: OCRDocument,
    max_candidates: int = 10,
) -> list[ExtractionCandidate]:
    candidates: list[ExtractionCandidate] = []
    for group in generate_text_groups(document, max_items=3):
        compact = compact_alphanumeric(group.text)
        if len(compact) < 10 or len(compact) > 32:
            continue
        for raw_window in _windows(compact):
            coerced = _coerce_candidate(raw_window)
            if coerced is None:
                continue
            observed, corrections = coerced

            if len(observed) == 10:
                try:
                    suggested_value = append_check_digit(observed)
                except ValueError:
                    continue
                value = observed
                validation = "inferred"
                verification = "unverified"
                observed_check_digit = None
                calculated_check_digit = suggested_value[-1]
                check_digit_source = "calculated"
                inferred = True
                rule_score = 0.70
            else:
                value = observed
                calculated_value = append_check_digit(observed[:10])
                suggested_value = calculated_value
                observed_check_digit = observed[-1]
                calculated_check_digit = calculated_value[-1]
                check_digit_source = "observed"
                inferred = False
                if validate_container_number(value):
                    validation = "valid"
                    verification = "verified"
                    rule_score = 1.0
                else:
                    validation = "invalid"
                    verification = "mismatch"
                    rule_score = 0.30

            extraction_score = clamp_score(
                group.confidence * 0.62
                + rule_score * 0.33
                + 0.05
                - corrections * 0.055
                - max(0, len(group.items) - 2) * 0.01
            )
            candidates.append(
                ExtractionCandidate(
                    value=value,
                    observed_value=observed,
                    ocr_confidence=round(group.confidence, 6),
                    extraction_score=extraction_score,
                    validation=validation,
                    inferred=inferred,
                    corrections=corrections,
                    source_texts=group.source_texts,
                    source_indices=group.source_indices,
                    boxes=group.boxes,
                    suggested_value=suggested_value,
                    verification=verification,
                    observed_check_digit=observed_check_digit,
                    calculated_check_digit=calculated_check_digit,
                    check_digit_source=check_digit_source,
                )
            )
    return deduplicate_candidates(candidates, max_candidates=max_candidates)
