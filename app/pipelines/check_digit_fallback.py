from __future__ import annotations

import re
from io import BytesIO
from typing import Any, Protocol

from PIL import Image, ImageOps

from app.pipelines.common import ExtractionCandidate, clamp_score
from app.validators.iso6346 import append_check_digit, validate_container_number


VARIANTS = ("boxed_color", "boxed_gray", "inner_gray")


class CheckDigitRecognizer(Protocol):
    model_name: str
    device: str

    def recognize(self, images: dict[str, Image.Image]) -> dict[str, dict[str, Any]]: ...


def normalize_single_digit(value: Any) -> str | None:
    compact = re.sub(r"[^A-Z0-9]", "", str(value or "").upper())
    return compact if len(compact) == 1 and compact.isdigit() else None


def _smooth(values: list[float], radius: int = 2) -> list[float]:
    output: list[float] = []
    for index in range(len(values)):
        start = max(0, index - radius)
        end = min(len(values), index + radius + 1)
        output.append(sum(values[start:end]) / (end - start))
    return output


def _strongest_pair(
    values: list[float],
    minimum_gap: int,
    maximum_gap: int,
    center_minimum: float,
    center_maximum: float,
) -> tuple[int, int] | None:
    ranked = sorted(
        range(len(values)), key=lambda index: values[index], reverse=True
    )[:40]
    best: tuple[float, int, int] | None = None
    for first in ranked:
        for second in ranked:
            if second <= first:
                continue
            gap = second - first
            center = (first + second) / 2
            if not minimum_gap <= gap <= maximum_gap:
                continue
            if not center_minimum <= center <= center_maximum:
                continue
            candidate = (values[first] + values[second], first, second)
            if best is None or candidate > best:
                best = candidate
    return None if best is None else (best[1], best[2])


def crop_inside_check_digit_box(
    image: Image.Image,
) -> tuple[Image.Image, dict[str, Any]]:
    """Remove artificial padding and, when detectable, the printed digit box."""

    source = image.convert("RGB")
    source_width, source_height = source.size
    pad_x = int(source_width * 0.115)
    pad_y = int(source_height * 0.115)
    if source_width - pad_x * 2 < 3 or source_height - pad_y * 2 < 3:
        content = source
    else:
        content = source.crop(
            (pad_x, pad_y, source_width - pad_x, source_height - pad_y)
        )
    gray = ImageOps.autocontrast(ImageOps.grayscale(content))
    pixels = gray.load()
    width, height = gray.size
    if width < 8 or height < 8:
        return gray.convert("RGB"), {"box_detected": False}

    y_start, y_end = int(height * 0.08), max(1, int(height * 0.92))
    columns: list[float] = []
    for x in range(1, width - 1):
        differences = [
            abs(pixels[x + 1, y] - pixels[x - 1, y])
            for y in range(y_start, y_end)
        ]
        columns.append(
            float(sum(differences) + sum(value > 24 for value in differences) * 50)
        )
    columns = [0.0, *_smooth(columns), 0.0]
    vertical = _strongest_pair(
        columns,
        max(2, int(width * 0.18)),
        max(3, int(width * 0.50)),
        width * 0.25,
        width * 0.68,
    )
    if vertical is None:
        return gray.convert("RGB"), {"box_detected": False}
    left, right = vertical

    x_start = max(1, left - int(width * 0.03))
    x_end = min(width - 1, right + int(width * 0.03))
    rows: list[float] = []
    for y in range(1, height - 1):
        differences = [
            abs(pixels[x, y + 1] - pixels[x, y - 1])
            for x in range(x_start, x_end)
        ]
        rows.append(
            float(sum(differences) + sum(value > 24 for value in differences) * 50)
        )
    rows = [0.0, *_smooth(rows), 0.0]
    horizontal = _strongest_pair(
        rows,
        max(2, int(height * 0.35)),
        max(3, int(height * 0.90)),
        height * 0.30,
        height * 0.70,
    )
    if horizontal is None:
        return gray.convert("RGB"), {"box_detected": False}
    top, bottom = horizontal

    inset_x = max(2, int((right - left) * 0.06))
    inset_y = max(2, int((bottom - top) * 0.05))
    inner_box = (
        min(right - 1, left + inset_x),
        min(bottom - 1, top + inset_y),
        max(left + 1, right - inset_x),
        max(top + 1, bottom - inset_y),
    )
    if inner_box[2] <= inner_box[0] or inner_box[3] <= inner_box[1]:
        return gray.convert("RGB"), {"box_detected": False}

    inner = ImageOps.autocontrast(ImageOps.grayscale(content.crop(inner_box)))
    corners = (
        (0, 0),
        (width - 1, 0),
        (0, height - 1),
        (width - 1, height - 1),
    )
    background = int(sum(gray.getpixel(point) for point in corners) / len(corners))
    border_x = max(8, int(inner.width * 0.35))
    border_y = max(8, int(inner.height * 0.20))
    inner = ImageOps.expand(
        inner,
        border=(border_x, border_y, border_x, border_y),
        fill=background,
    )
    return inner.convert("RGB"), {
        "box_detected": True,
        "detected_box": [left, top, right, bottom],
        "inner_box": list(inner_box),
    }


def create_check_digit_variants(
    image_bytes: bytes,
    candidate: ExtractionCandidate,
) -> tuple[dict[str, Image.Image], dict[str, Any]]:
    if len(candidate.observed_value) != 10 or not candidate.boxes:
        raise ValueError("check-digit fallback requires a boxed 10-character candidate")
    with Image.open(BytesIO(image_bytes)) as opened:
        image = ImageOps.exif_transpose(opened).convert("RGB")
    image_width, image_height = image.size
    left = min(float(box[0]) for box in candidate.boxes)
    top = min(float(box[1]) for box in candidate.boxes)
    right = max(float(box[2]) for box in candidate.boxes)
    bottom = max(float(box[3]) for box in candidate.boxes)
    text_width = max(1.0, right - left)
    text_height = max(1.0, bottom - top)
    vertical_padding = text_height * 0.25
    crop_box = (
        max(0, int(right)),
        max(0, int(top - vertical_padding)),
        min(image_width, int(right + text_width * 0.30)),
        min(image_height, int(bottom + vertical_padding)),
    )
    if crop_box[2] <= crop_box[0] or crop_box[3] <= crop_box[1]:
        raise ValueError("calculated check-digit crop is empty")

    cropped = image.crop(crop_box)
    corners = (
        cropped.getpixel((0, 0)),
        cropped.getpixel((max(0, cropped.width - 1), 0)),
        cropped.getpixel((0, max(0, cropped.height - 1))),
        cropped.getpixel((max(0, cropped.width - 1), max(0, cropped.height - 1))),
    )
    background = tuple(
        sum(int(pixel[channel]) for pixel in corners) // len(corners)
        for channel in range(3)
    )
    border_x = max(4, int(cropped.width * 0.15))
    border_y = max(4, int(cropped.height * 0.15))
    cropped = ImageOps.expand(
        cropped,
        border=(border_x, border_y, border_x, border_y),
        fill=background,
    )
    cropped = cropped.resize(
        (cropped.width * 4, cropped.height * 4),
        Image.Resampling.LANCZOS,
    )
    inner, inner_metadata = crop_inside_check_digit_box(cropped)
    variants = {
        "boxed_color": cropped.copy(),
        "boxed_gray": ImageOps.autocontrast(ImageOps.grayscale(cropped)).convert("RGB"),
        "inner_gray": inner,
    }
    return variants, {
        "source_candidate": candidate.observed_value,
        "source_boxes": [list(box) for box in candidate.boxes],
        "crop_box": list(crop_box),
        "inner_gray": inner_metadata,
    }


def select_digit_without_truth(
    recognitions: dict[str, dict[str, Any]],
    *,
    minimum_candidate_score: float,
    minimum_single_score: float,
) -> dict[str, Any]:
    eligible: list[tuple[str, str, float]] = []
    for variant, result in recognitions.items():
        digit = normalize_single_digit(result.get("text"))
        score = float(result.get("score") or 0.0)
        if digit is not None and score >= minimum_candidate_score:
            eligible.append((variant, digit, score))
    if not eligible:
        return {
            "status": "not_found",
            "digit": None,
            "reason": "no_single_digit_above_threshold",
            "supporting_variants": [],
            "score": 0.0,
        }

    by_digit: dict[str, list[tuple[str, float]]] = {}
    for variant, digit, score in eligible:
        by_digit.setdefault(digit, []).append((variant, score))
    ranking = sorted(
        by_digit.items(),
        key=lambda item: (len(item[1]), max(score for _variant, score in item[1])),
        reverse=True,
    )
    best_digit, best_support = ranking[0]
    best_count = len(best_support)
    tied = len(ranking) > 1 and len(ranking[1][1]) == best_count
    best_score = max(score for _variant, score in best_support)

    if best_count >= 2 and not tied:
        return {
            "status": "selected",
            "digit": best_digit,
            "reason": "variant_consensus",
            "supporting_variants": [variant for variant, _score in best_support],
            "score": round(best_score, 6),
        }
    if len(by_digit) == 1 and best_score >= minimum_single_score:
        return {
            "status": "selected",
            "digit": best_digit,
            "reason": "high_confidence_single_result",
            "supporting_variants": [variant for variant, _score in best_support],
            "score": round(best_score, 6),
        }
    return {
        "status": "conflict" if len(by_digit) > 1 else "low_confidence",
        "digit": None,
        "reason": (
            "different_digits_between_variants"
            if len(by_digit) > 1
            else "single_result_below_auto_select_threshold"
        ),
        "supporting_variants": [],
        "score": round(best_score, 6),
        "candidates": {
            digit: [
                {"variant": variant, "score": round(score, 6)}
                for variant, score in support
            ]
            for digit, support in sorted(by_digit.items())
        },
    }


def apply_check_digit_fallback(
    image_bytes: bytes,
    candidate: ExtractionCandidate,
    recognizer: CheckDigitRecognizer,
    *,
    minimum_candidate_score: float,
    minimum_single_score: float,
) -> tuple[ExtractionCandidate | None, dict[str, Any]]:
    variants, preprocessing = create_check_digit_variants(image_bytes, candidate)
    recognitions = recognizer.recognize(variants)
    for result in recognitions.values():
        result["digit"] = normalize_single_digit(result.get("text"))
    selection = select_digit_without_truth(
        recognitions,
        minimum_candidate_score=minimum_candidate_score,
        minimum_single_score=minimum_single_score,
    )
    metadata = {
        "status": selection["status"],
        "strategy": "english_check_digit_direct_recognition",
        "model_name": recognizer.model_name,
        "device": recognizer.device,
        "variants": recognitions,
        "selection": selection,
        "preprocessing": preprocessing,
    }
    observed_digit = selection.get("digit")
    if observed_digit is None:
        return None, metadata

    observed = f"{candidate.observed_value}{observed_digit}"
    suggested = append_check_digit(candidate.observed_value)
    verified = validate_container_number(observed)
    verification = "verified" if verified else "mismatch"
    validation = "valid" if verified else "invalid"
    digit_confidence = float(selection.get("score") or 0.0)
    confidence = round((candidate.ocr_confidence + digit_confidence) / 2, 6)
    rule_score = 1.0 if verified else 0.30
    extraction_score = clamp_score(
        confidence * 0.62
        + rule_score * 0.33
        + 0.05
        - candidate.corrections * 0.055
    )
    completed = ExtractionCandidate(
        value=observed,
        observed_value=observed,
        suggested_value=suggested,
        ocr_confidence=confidence,
        extraction_score=extraction_score,
        validation=validation,
        verification=verification,
        observed_check_digit=observed_digit,
        calculated_check_digit=suggested[-1],
        check_digit_source="observed",
        inferred=False,
        corrections=candidate.corrections,
        source_texts=candidate.source_texts,
        source_indices=candidate.source_indices,
        boxes=candidate.boxes,
        orientation=candidate.orientation,
        supporting_orientations=candidate.supporting_orientations,
        selection_reasons=(
            *candidate.selection_reasons,
            "check_digit_direct_recognition",
            str(selection["reason"]),
        ),
    )
    metadata.update(
        {
            "status": "applied",
            "observed_value": observed,
            "verification": verification,
            "calculated_check_digit": suggested[-1],
        }
    )
    return completed, metadata
