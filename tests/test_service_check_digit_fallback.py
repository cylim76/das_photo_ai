from io import BytesIO

from PIL import Image

from app.config import Settings
from app.domain import OCRDocument, OCRItem
from app.service import recognize_image


def uploaded_image() -> bytes:
    buffer = BytesIO()
    Image.new("RGB", (500, 300), "brown").save(buffer, format="PNG")
    return buffer.getvalue()


def document(check_digit: str = "") -> OCRDocument:
    return OCRDocument(
        items=(
            OCRItem(
                text="HASU",
                score=0.99,
                box=(50, 100, 150, 140),
                source_index=0,
            ),
            OCRItem(
                text=f"506039{check_digit}",
                score=0.98,
                box=(160, 100, 300, 140),
                source_index=1,
            ),
        ),
        source="test:paddle",
    )


class FakeEngine:
    name = "paddle_gpu"

    def __init__(self, source: OCRDocument) -> None:
        self.source = source

    def recognize(self, _input_data):
        return self.source


class FakeRecognizer:
    model_name = "en_PP-OCRv5_mobile_rec"
    device = "gpu:0"

    def recognize(self, _images):
        return {
            "boxed_color": {"text": "6", "score": 0.81},
            "boxed_gray": {"text": "6", "score": 0.91},
            "inner_gray": {"text": "6", "score": 0.99},
        }


def test_image_service_applies_check_digit_fallback(monkeypatch) -> None:
    monkeypatch.setattr(
        "app.service.create_engine",
        lambda *_args: FakeEngine(document()),
    )
    monkeypatch.setattr(
        "app.service.create_check_digit_recognizer",
        lambda *_args: FakeRecognizer(),
    )

    response = recognize_image(
        image_bytes=uploaded_image(),
        image_filename="sample.png",
        targets=["container_number"],
        engine_name="paddle_gpu",
        request_id="check-digit-test",
        settings=Settings(container_check_digit_fallback=True),
    )

    result = response.results["container_number"]
    assert result.observed_value == "HASU5060396"
    assert result.verification == "verified"
    assert result.observed_check_digit == "6"
    assert result.postprocessing["status"] == "applied"
    assert result.postprocessing["model_name"] == "en_PP-OCRv5_mobile_rec"
    assert "check_digit_direct_recognition" in result.candidates[0].selection_reasons


def test_image_service_does_not_run_fallback_for_verified_number(monkeypatch) -> None:
    monkeypatch.setattr(
        "app.service.create_engine",
        lambda *_args: FakeEngine(document("6")),
    )

    def unexpected(*_args):
        raise AssertionError("check-digit recognizer should not be created")

    monkeypatch.setattr("app.service.create_check_digit_recognizer", unexpected)
    response = recognize_image(
        image_bytes=uploaded_image(),
        image_filename="sample.png",
        targets=["container_number"],
        engine_name="paddle_gpu",
        request_id=None,
        settings=Settings(container_check_digit_fallback=True),
    )

    result = response.results["container_number"]
    assert result.observed_value == "HASU5060396"
    assert result.postprocessing["status"] == "not_needed"


def test_optional_fallback_error_keeps_primary_result(monkeypatch) -> None:
    monkeypatch.setattr(
        "app.service.create_engine",
        lambda *_args: FakeEngine(document()),
    )

    def unavailable(*_args):
        raise RuntimeError("test model unavailable")

    monkeypatch.setattr("app.service.create_check_digit_recognizer", unavailable)
    response = recognize_image(
        image_bytes=uploaded_image(),
        image_filename="sample.png",
        targets=["container_number"],
        engine_name="paddle_gpu",
        request_id=None,
        settings=Settings(container_check_digit_fallback=True),
    )

    result = response.results["container_number"]
    assert result.observed_value == "HASU506039"
    assert result.verification == "unverified"
    assert result.postprocessing["status"] == "error"
