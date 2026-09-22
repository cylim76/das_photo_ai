from pathlib import Path

from PIL import Image

from scripts.evaluate_container_check_digit import (
    combine_isolated_check_digit,
    create_container_variant,
    load_retry_file_names,
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


def test_right_30_variant_extends_only_the_right_edge(tmp_path: Path) -> None:
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
        "right_30_2x",
        tmp_path / "right-30.png",
    )

    assert output is not None
    with Image.open(output) as cropped:
        # Original observed band is 220x30; 30% is added only to the right.
        assert cropped.size == (572, 60)


def test_v032_variants_add_vertical_context_and_isolate_check_digit(
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

    padded = create_container_variant(
        source,
        original,
        "right_30_vpad15_2x",
        tmp_path / "padded.png",
    )
    digit = create_container_variant(
        source,
        original,
        "check_digit_strip_4x",
        tmp_path / "digit.png",
    )

    assert padded is not None
    assert digit is not None
    with Image.open(padded) as cropped:
        assert cropped.size == (572, 78)
    with Image.open(digit) as cropped:
        assert cropped.size == (336, 228)


def test_combines_only_an_observed_isolated_digit() -> None:
    prefix = payload(
        "HASU506039",
        "unverified",
        0.8,
        [[0, 0, 80, 30], [90, 0, 220, 30]],
    )
    prefix_candidate = prefix["results"]["container_number"]["candidates"][0]
    prefix_candidate.update(
        {
            "ocr_confidence": 0.95,
            "source_texts": ["HASU", "506039"],
            "source_indices": [0, 1],
        }
    )
    digit_payload = {
        "ocr_items": [
            {
                "text": "6",
                "score": 0.91,
                "source_index": 0,
                "box": [5, 5, 25, 35],
            }
        ],
        "results": {"container_number": {}},
    }

    combined = combine_isolated_check_digit(
        prefix,
        digit_payload,
        digit_is_separate_crop=True,
        strategy="test_digit_crop",
    )

    result = combined["results"]["container_number"]
    assert result["observed_value"] == "HASU5060396"
    assert result["observed_check_digit"] == "6"
    assert result["verification"] == "verified"
    assert combined["postprocessing"]["strategy"] == "test_digit_crop"


def test_does_not_use_calculated_digit_when_ocr_observed_none() -> None:
    prefix = payload(
        "HASU506039",
        "unverified",
        0.8,
        [[0, 0, 80, 30], [90, 0, 220, 30]],
    )
    digit_payload = {
        "ocr_items": [{"text": "BOX", "score": 0.99, "source_index": 0}],
        "results": {"container_number": {}},
    }

    combined = combine_isolated_check_digit(
        prefix,
        digit_payload,
        digit_is_separate_crop=True,
        strategy="test_digit_crop",
    )

    assert combined == digit_payload


def test_ignores_low_confidence_isolated_digit() -> None:
    prefix = payload(
        "HASU506039",
        "unverified",
        0.8,
        [[0, 0, 80, 30], [90, 0, 220, 30]],
    )
    digit_payload = {
        "ocr_items": [
            {
                "text": "6",
                "score": 0.20,
                "source_index": 0,
                "box": [5, 5, 25, 35],
            }
        ],
        "results": {"container_number": {}},
    }

    combined = combine_isolated_check_digit(
        prefix,
        digit_payload,
        digit_is_separate_crop=True,
        strategy="test_digit_crop",
    )

    assert combined == digit_payload


def test_associates_right_side_digit_observed_in_same_crop() -> None:
    crop_payload = payload(
        "TGBU491605",
        "unverified",
        0.8,
        [[3, 19, 189, 115], [251, 3, 515, 103]],
    )
    candidate = crop_payload["results"]["container_number"]["candidates"][0]
    candidate.update(
        {
            "ocr_confidence": 0.99,
            "source_texts": ["TGBU", "491605"],
            "source_indices": [2, 0],
        }
    )
    crop_payload["ocr_items"] = [
        {
            "text": "491605",
            "score": 0.99,
            "source_index": 0,
            "box": [251, 3, 515, 103],
        },
        {
            "text": "0",
            "score": 0.89,
            "source_index": 1,
            "box": [574, 6, 617, 71],
        },
        {
            "text": "TGBU",
            "score": 0.99,
            "source_index": 2,
            "box": [3, 19, 189, 115],
        },
    ]

    combined = combine_isolated_check_digit(
        crop_payload,
        crop_payload,
        digit_is_separate_crop=False,
        strategy="right_side_isolated_digit_association",
    )

    result = combined["results"]["container_number"]
    assert result["observed_value"] == "TGBU4916050"
    assert result["verification"] == "verified"


def test_loads_retry_names_from_previous_summary(tmp_path: Path) -> None:
    summary = tmp_path / "summary.json"
    summary.write_text(
        '{"metrics":{"still_unverified_or_wrong":["A.jpeg","B.jpg"]}}',
        encoding="utf-8",
    )

    assert load_retry_file_names(summary) == {"A.jpeg", "B.jpg"}


def test_selection_prefers_observed_verified_value_without_truth() -> None:
    selected, _result = select_without_truth(
        {
            "original": payload("HASU506039", "unverified", 0.95),
            "context_2x": payload("HASU5060396", "verified", 0.80),
            "context_4x": payload("HASU5060394", "mismatch", 0.99),
        }
    )

    assert selected == "context_2x"
