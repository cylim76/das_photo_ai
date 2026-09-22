from app.config import Settings
from app.domain import OCRDocument, OCRItem
from app.service import recognize_image


def document(text: str, orientation: str) -> OCRDocument:
    return OCRDocument(
        items=(OCRItem(text=text, score=0.99, source_index=0),),
        source="test",
        metadata={"orientation": orientation},
    )


def test_image_service_fuses_seal_orientations(monkeypatch) -> None:
    class FakeEngine:
        name = "paddle_gpu"

        def recognize_orientations(self, _input_data):
            return {
                "original": document("1000508518", "original"),
                "cw90": document("MLCN4062676", "cw90"),
                "ccw90": document("110402120220", "ccw90"),
            }

    monkeypatch.setattr("app.service.create_engine", lambda *_args: FakeEngine())

    response = recognize_image(
        image_bytes=b"fake-image",
        image_filename="sample.jpeg",
        targets=["seal_number"],
        engine_name="paddle_gpu",
        request_id="orientation-test",
        settings=Settings(),
    )

    result = response.results["seal_number"]
    assert result.value == "MLCN4062676"
    assert result.orientation == "cw90"
    assert result.candidates[0].orientation == "cw90"
