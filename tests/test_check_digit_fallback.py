from io import BytesIO

from PIL import Image, ImageDraw

from app.pipelines.check_digit_fallback import (
    apply_check_digit_fallback,
    crop_inside_check_digit_box,
    normalize_single_digit,
    select_digit_without_truth,
)
from app.pipelines.common import ExtractionCandidate


def source_candidate() -> ExtractionCandidate:
    return ExtractionCandidate(
        value="HASU506039",
        observed_value="HASU506039",
        suggested_value="HASU5060396",
        ocr_confidence=0.96,
        extraction_score=0.88,
        validation="inferred",
        verification="unverified",
        observed_check_digit=None,
        calculated_check_digit="6",
        check_digit_source="calculated",
        inferred=True,
        corrections=0,
        source_texts=("HASU", "506039"),
        source_indices=(0, 1),
        boxes=((50, 100, 150, 140), (160, 100, 300, 140)),
    )


def image_bytes() -> bytes:
    buffer = BytesIO()
    image = Image.new("RGB", (500, 300), "brown")
    draw = ImageDraw.Draw(image)
    draw.rectangle((320, 90, 365, 155), outline="white", width=4)
    draw.text((335, 105), "6", fill="white")
    image.save(buffer, format="PNG")
    return buffer.getvalue()


class FakeRecognizer:
    model_name = "en_PP-OCRv5_mobile_rec"
    device = "gpu:0"

    def __init__(self, results: dict[str, dict]) -> None:
        self.results = results
        self.seen_variants: set[str] = set()

    def recognize(self, images):
        self.seen_variants = set(images)
        return {name: dict(result) for name, result in self.results.items()}


def test_normalize_single_digit_does_not_map_letters() -> None:
    assert normalize_single_digit("6") == "6"
    assert normalize_single_digit(" I ") is None
    assert normalize_single_digit("间") is None


def test_select_digit_uses_consensus_without_iso() -> None:
    selection = select_digit_without_truth(
        {
            "boxed_color": {"text": "6", "score": 0.60},
            "boxed_gray": {"text": "6", "score": 0.80},
            "inner_gray": {"text": "8", "score": 0.99},
        },
        minimum_candidate_score=0.50,
        minimum_single_score=0.70,
    )

    assert selection["digit"] == "6"
    assert selection["reason"] == "variant_consensus"


def test_check_digit_fallback_builds_verified_observed_candidate() -> None:
    recognizer = FakeRecognizer(
        {
            "boxed_color": {"text": "间", "score": 0.3},
            "boxed_gray": {"text": "6", "score": 0.85},
            "inner_gray": {"text": "6", "score": 0.99},
        }
    )

    completed, metadata = apply_check_digit_fallback(
        image_bytes(),
        source_candidate(),
        recognizer,
        minimum_candidate_score=0.50,
        minimum_single_score=0.70,
    )

    assert recognizer.seen_variants == {"boxed_color", "boxed_gray", "inner_gray"}
    assert completed is not None
    assert completed.observed_value == "HASU5060396"
    assert completed.observed_check_digit == "6"
    assert completed.check_digit_source == "observed"
    assert completed.verification == "verified"
    assert completed.inferred is False
    assert metadata["status"] == "applied"
    assert metadata["selection"]["reason"] == "variant_consensus"


def test_check_digit_fallback_preserves_observed_mismatch() -> None:
    recognizer = FakeRecognizer(
        {
            "boxed_color": {"text": "4", "score": 0.90},
            "boxed_gray": {"text": "4", "score": 0.92},
            "inner_gray": {"text": "I", "score": 0.99},
        }
    )

    completed, metadata = apply_check_digit_fallback(
        image_bytes(),
        source_candidate(),
        recognizer,
        minimum_candidate_score=0.50,
        minimum_single_score=0.70,
    )

    assert completed is not None
    assert completed.observed_value == "HASU5060394"
    assert completed.suggested_value == "HASU5060396"
    assert completed.verification == "mismatch"
    assert metadata["verification"] == "mismatch"


def test_check_digit_fallback_does_not_choose_conflicting_digits() -> None:
    recognizer = FakeRecognizer(
        {
            "boxed_color": {"text": "1", "score": 0.91},
            "boxed_gray": {"text": "7", "score": 0.92},
            "inner_gray": {"text": "", "score": 0.0},
        }
    )

    completed, metadata = apply_check_digit_fallback(
        image_bytes(),
        source_candidate(),
        recognizer,
        minimum_candidate_score=0.50,
        minimum_single_score=0.70,
    )

    assert completed is None
    assert metadata["status"] == "conflict"


def test_crop_inside_box_keeps_an_image_when_border_is_broken() -> None:
    image = Image.new("RGB", (320, 360), (70, 70, 70))
    draw = ImageDraw.Draw(image)
    draw.line((110, 70, 190, 70), fill="white", width=8)
    draw.line((210, 70, 210, 290), fill="white", width=8)
    draw.line((110, 290, 210, 290), fill="white", width=8)
    draw.line((110, 70, 110, 145), fill="white", width=8)
    draw.line((110, 180, 110, 290), fill="white", width=8)
    draw.line((150, 115, 180, 115), fill="white", width=12)
    draw.line((180, 115, 158, 255), fill="white", width=12)

    cropped, metadata = crop_inside_check_digit_box(image)

    assert cropped.width > 0
    assert cropped.height > 0
    assert "box_detected" in metadata
