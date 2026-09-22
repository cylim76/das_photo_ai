import sys
import types
from pathlib import Path

from PIL import Image

from app.engines.check_digit_recognizer import (
    PaddleCheckDigitRecognizer,
    clear_check_digit_recognizer_cache,
    create_check_digit_recognizer,
    is_check_digit_recognizer_loaded,
)


def test_recognition_only_model_uses_named_model_and_device(monkeypatch) -> None:
    state: dict[str, object] = {}

    class FakeResult:
        json = {"res": {"rec_text": "6", "rec_score": 0.987}}

    class FakeTextRecognition:
        def __init__(self, **kwargs) -> None:
            state["kwargs"] = kwargs

        def predict(self, image_path: str, batch_size: int):
            state["image_exists"] = Path(image_path).is_file()
            state["batch_size"] = batch_size
            return [FakeResult()]

    module = types.ModuleType("paddleocr")
    module.TextRecognition = FakeTextRecognition  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "paddleocr", module)

    recognizer = PaddleCheckDigitRecognizer("en_PP-OCRv5_mobile_rec", "gpu:0")
    results = recognizer.recognize(
        {"inner_gray": Image.new("RGB", (80, 120), "white")}
    )

    assert state["kwargs"] == {
        "model_name": "en_PP-OCRv5_mobile_rec",
        "device": "gpu:0",
    }
    assert state["image_exists"] is True
    assert state["batch_size"] == 1
    assert results["inner_gray"]["text"] == "6"
    assert results["inner_gray"]["score"] == 0.987


def test_check_digit_factory_reuses_model(monkeypatch) -> None:
    state = {"initializations": 0}

    class FakeTextRecognition:
        def __init__(self, **_kwargs) -> None:
            state["initializations"] += 1

    module = types.ModuleType("paddleocr")
    module.TextRecognition = FakeTextRecognition  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "paddleocr", module)

    clear_check_digit_recognizer_cache()
    try:
        first = create_check_digit_recognizer("en_PP-OCRv5_mobile_rec", "gpu:0")
        second = create_check_digit_recognizer("en_PP-OCRv5_mobile_rec", "gpu:0")
        loaded = is_check_digit_recognizer_loaded(
            "en_PP-OCRv5_mobile_rec",
            "gpu:0",
        )
    finally:
        clear_check_digit_recognizer_cache()

    assert first is second
    assert state["initializations"] == 1
    assert loaded is True
