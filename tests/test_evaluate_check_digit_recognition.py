from pathlib import Path

from PIL import Image, ImageDraw

from scripts.evaluate_check_digit_recognition import (
    _result_payload,
    crop_inside_check_digit_box,
    create_recognition_variants,
    normalize_single_digit,
    select_digit_without_truth,
    summarize_records,
)


def result(text: str, score: float) -> dict:
    return {"text": text, "score": score}


def test_result_payload_unwraps_paddle_result() -> None:
    assert _result_payload({"res": {"rec_text": "7", "rec_score": 0.98}}) == {
        "rec_text": "7",
        "rec_score": 0.98,
    }


def test_normalize_single_digit_is_strict() -> None:
    assert normalize_single_digit(" 7 ") == "7"
    assert normalize_single_digit("[7]") == "7"
    assert normalize_single_digit("I") is None
    assert normalize_single_digit("17") is None
    assert normalize_single_digit("") is None


def test_selection_uses_consensus_without_iso_or_truth() -> None:
    selected = select_digit_without_truth(
        {
            "boxed_color": result("7", 0.62),
            "boxed_gray": result("7", 0.81),
            "inner_gray": result("1", 0.99),
        }
    )

    assert selected["status"] == "selected"
    assert selected["digit"] == "7"
    assert selected["reason"] == "variant_consensus"


def test_selection_accepts_one_high_confidence_digit() -> None:
    selected = select_digit_without_truth(
        {
            "boxed_color": result("BOX", 0.99),
            "boxed_gray": result("", 0.0),
            "inner_gray": result("6", 0.92),
        }
    )

    assert selected["status"] == "selected"
    assert selected["digit"] == "6"
    assert selected["reason"] == "high_confidence_single_result"


def test_selection_preserves_conflict_for_manual_review() -> None:
    selected = select_digit_without_truth(
        {
            "boxed_color": result("1", 0.94),
            "boxed_gray": result("7", 0.95),
            "inner_gray": result("", 0.0),
        }
    )

    assert selected["status"] == "conflict"
    assert selected["digit"] is None
    assert set(selected["candidates"]) == {"1", "7"}


def test_selection_rejects_low_confidence_single_result() -> None:
    selected = select_digit_without_truth(
        {
            "boxed_color": result("", 0.0),
            "boxed_gray": result("1", 0.60),
            "inner_gray": result("TEXT", 0.99),
        }
    )

    assert selected["status"] == "low_confidence"
    assert selected["digit"] is None


def test_inner_crop_and_saved_variants_do_not_require_expected_digit(
    tmp_path: Path,
) -> None:
    source = tmp_path / "digit.png"
    image = Image.new("RGB", (320, 360), (70, 70, 70))
    draw = ImageDraw.Draw(image)
    draw.rectangle((110, 70, 210, 290), outline="white", width=10)
    draw.line((150, 110, 180, 110), fill="white", width=12)
    draw.line((180, 110, 155, 250), fill="white", width=12)
    image.save(source)

    cropped, metadata = crop_inside_check_digit_box(image)
    variants = create_recognition_variants(source, tmp_path / "variants", "sample")

    assert cropped.width > 0 and cropped.height > 0
    assert set(variants) == {"boxed_color", "boxed_gray", "inner_gray"}
    assert all(item["path"].is_file() for item in variants.values())
    assert "box_detected" in metadata


def test_summary_keeps_false_verified_visible() -> None:
    records = [
        {
            "file_name": "A.jpeg",
            "selection_status": "selected",
            "selection_reason": "variant_consensus",
            "verification": "verified",
            "selected_verified_exact": True,
            "recognitions": {
                "boxed_color": {"digit": "7", "digit_exact": True, "score": 0.9}
            },
        },
        {
            "file_name": "B.jpeg",
            "selection_status": "selected",
            "selection_reason": "high_confidence_single_result",
            "verification": "verified",
            "selected_verified_exact": False,
            "recognitions": {
                "boxed_color": {"digit": "1", "digit_exact": False, "score": 0.8}
            },
        },
    ]

    metrics = summarize_records(
        records,
        {"metrics": {"evaluated": 12, "selected_verified_exact": 6}},
    )

    assert metrics["direct_recognition_rescued"] == 1
    assert metrics["selected_false_verified"] == 1
    assert metrics["previous_run_scope"]["combined_verified"] == 7
    assert metrics["still_unverified_or_wrong"] == ["B.jpeg"]
