from pathlib import Path

from PIL import Image

from scripts.evaluate_container_check_digit import (
    create_container_variant,
    select_without_truth,
)


def payload(
    observed: str,
    verification: str,
    score: float,
    boxes: list[list[float]] | None = None,
) -> dict:
    candidate = {
        "observed_value": observed,
        "boxes": boxes or [],
    }
    return {
        "results": {
            "container_number": {
                "observed_value": observed,
                "verification": verification,
                "extraction_score": score,
                "ocr_confidence": 0.9,
                "candidates": [candidate],
            }
        }
    }


def test_create_container_variant_includes_space_right_of_observed_number(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.png"
    Image.new("RGB", (500, 300), "white").save(source)
    original = payload(
        "HASU506039",
        "unverified",
        0.8,
        [[100, 100, 180, 130], [190, 100, 320, 130]],
    )

    output = create_container_variant(
        source,
        original,
        "context_2x",
        tmp_path / "crop.png",
    )

    assert output is not None
    with Image.open(output) as cropped:
        assert cropped.width > (320 - 100) * 2
        assert cropped.height > (130 - 100) * 2


def test_selection_prefers_observed_verified_value_without_truth() -> None:
    selected, _result = select_without_truth(
        {
            "original": payload("HASU506039", "unverified", 0.95),
            "context_2x": payload("HASU5060396", "verified", 0.80),
            "context_4x": payload("HASU5060394", "mismatch", 0.99),
        }
    )

    assert selected == "context_2x"
