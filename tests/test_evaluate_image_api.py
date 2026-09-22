from pathlib import Path

from scripts.evaluate_image_api import _image_files, evaluate_payload, summarize


def test_image_files_excludes_rendered_ocr_images(tmp_path: Path) -> None:
    original = tmp_path / "F119755.jpeg"
    rendered = tmp_path / "F119755_ocr_res_img.jpeg"
    json_file = tmp_path / "F119755_res.json"
    for path in (original, rendered, json_file):
        path.write_bytes(b"x")

    assert _image_files(tmp_path) == [original]


def test_evaluate_container_keeps_observed_and_suggested_separate() -> None:
    payload = {
        "service_version": "0.2.0",
        "extractor_version": "0.2.0",
        "engine": "paddle_gpu",
        "elapsed_ms": 530.683,
        "ocr_source": "paddleocr:gpu:0",
        "ocr_items": [{"text": "HASU"}, {"text": "506039"}],
        "results": {
            "container_number": {
                "status": "found",
                "observed_value": "HASU506039",
                "suggested_value": "HASU5060396",
                "verification": "unverified",
                "validation": "inferred",
                "postprocessing": {
                    "status": "not_found",
                    "strategy": "english_check_digit_direct_recognition",
                    "model_name": "en_PP-OCRv5_mobile_rec",
                },
                "candidates": [
                    {
                        "observed_value": "HASU506039",
                        "suggested_value": "HASU5060396",
                    }
                ],
            }
        },
    }

    result = evaluate_payload(payload, "container_number", "HASU5060396")

    assert result["outcome"] == "inferred_exact"
    assert result["observed_exact"] is False
    assert result["suggested_exact"] is True
    assert result["candidate_contains"] is True
    assert result["verification"] == "unverified"
    assert result["postprocessing_status"] == "not_found"
    assert result["postprocessing_model"] == "en_PP-OCRv5_mobile_rec"


def test_evaluate_seal_exact() -> None:
    payload = {
        "elapsed_ms": 100.0,
        "ocr_items": [{"text": "BX133"}],
        "results": {
            "seal_number": {
                "status": "found",
                "observed_value": "BX133",
                "verification": "not_applicable",
                "candidates": [{"value": "BX133"}],
            }
        },
    }

    result = evaluate_payload(payload, "seal_number", "BX133")

    assert result["outcome"] == "exact"
    assert result["observed_exact"] is True
    assert result["suggested_exact"] is True


def test_summary_reports_strict_and_suggested_accuracy() -> None:
    records = [
        {
            "target": "container_number",
            "api_success": True,
            "expected": "HASU5060396",
            "outcome": "inferred_exact",
            "observed_exact": False,
            "suggested_exact": True,
            "candidate_contains": True,
            "verification": "unverified",
            "postprocessing_status": "not_found",
            "service_elapsed_ms": 500.0,
        },
        {
            "target": "container_number",
            "api_success": True,
            "expected": "CSNU7084301",
            "outcome": "verified_exact",
            "observed_exact": True,
            "suggested_exact": True,
            "candidate_contains": True,
            "verification": "verified",
            "postprocessing_status": "applied",
            "service_elapsed_ms": 300.0,
        },
    ]

    result = summarize(records)["container_number"]

    assert result["evaluated"] == 2
    assert result["observed_top1_accuracy"] == 0.5
    assert result["verified_exact"] == 1
    assert result["false_verified"] == 0
    assert result["suggested_top1_accuracy"] == 1.0
    assert result["verification"] == {"unverified": 1, "verified": 1}
    assert result["postprocessing"] == {"applied": 1, "not_found": 1}
    assert result["service_elapsed_ms"]["median"] == 300.0


def test_summary_reports_false_verified_container_result() -> None:
    result = summarize(
        [
            {
                "target": "container_number",
                "api_success": True,
                "expected": "HASU5060396",
                "outcome": "wrong",
                "observed_exact": False,
                "suggested_exact": False,
                "candidate_contains": False,
                "verification": "verified",
                "service_elapsed_ms": 100.0,
            }
        ]
    )["container_number"]

    assert result["verified_exact"] == 0
    assert result["false_verified"] == 1
