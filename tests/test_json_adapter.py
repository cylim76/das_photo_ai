import pytest

from app.engines.base import EngineError
from app.engines.json_adapter import adapt_ocr_json


def test_adapts_paddleocr_parallel_arrays() -> None:
    document = adapt_ocr_json(
        {
            "input_path": "/tmp/photo.jpg",
            "rec_texts": ["HASU", "", "506039"],
            "rec_scores": [0.99, 0.0, 0.98],
            "rec_boxes": [[1, 2, 3, 4], [0, 0, 0, 0], [5, 6, 7, 8]],
        }
    )

    assert document.source == "paddleocr_json"
    assert [item.text for item in document.items] == ["HASU", "506039"]
    assert [item.source_index for item in document.items] == [0, 2]
    assert document.items[0].box == (1.0, 2.0, 3.0, 4.0)


def test_adapts_normalized_items_and_derives_box() -> None:
    document = adapt_ocr_json(
        {
            "items": [
                {
                    "text": "V542613",
                    "score": 0.999,
                    "polygon": [[10, 20], [30, 20], [30, 50], [10, 50]],
                }
            ]
        }
    )

    assert document.items[0].box == (10.0, 20.0, 30.0, 50.0)


def test_rejects_mismatched_paddle_arrays() -> None:
    with pytest.raises(EngineError):
        adapt_ocr_json({"rec_texts": ["ABC"], "rec_scores": []})


def test_unwraps_paddle_result_json_res_field() -> None:
    document = adapt_ocr_json(
        {"res": {"rec_texts": ["V542613"], "rec_scores": [0.99]}}
    )

    assert document.items[0].text == "V542613"
