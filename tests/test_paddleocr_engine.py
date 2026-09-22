import os
import sys
import types
from io import BytesIO
from pathlib import Path

from PIL import Image

from app.config import Settings
from app.domain import EngineInput
from app.engines.factory import clear_engine_cache, create_engine
from app.engines.paddleocr_engine import PaddleOcrEngine


def test_paddleocr_engine_uses_uploaded_image_and_normalizes_result(monkeypatch) -> None:
    state: dict[str, object] = {}

    class FakeResult:
        json = {
            "res": {
                "rec_texts": ["HASU", "506039"],
                "rec_scores": [0.99, 0.98],
                "rec_boxes": [[1, 2, 3, 4], [5, 6, 7, 8]],
            }
        }

    class FakePaddleOCR:
        def __init__(self, **kwargs) -> None:
            state["kwargs"] = kwargs

        def predict(self, image_path: str):
            state["image_path"] = image_path
            state["bytes"] = Path(image_path).read_bytes()
            return [FakeResult()]

    module = types.ModuleType("paddleocr")
    module.PaddleOCR = FakePaddleOCR  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "paddleocr", module)

    settings = Settings(
        paddle_device="gpu:0",
        paddle_lang="ch",
        paddle_ocr_version="PP-OCRv5",
    )
    engine = PaddleOcrEngine("paddle_gpu", settings)
    document = engine.recognize(
        EngineInput(image_bytes=b"image-data", image_filename="sample.jpeg")
    )

    assert state["bytes"] == b"image-data"
    assert state["kwargs"] == {
        "use_doc_orientation_classify": False,
        "use_doc_unwarping": False,
        "use_textline_orientation": False,
        "device": "gpu:0",
        "lang": "ch",
        "ocr_version": "PP-OCRv5",
    }
    assert [item.text for item in document.items] == ["HASU", "506039"]
    assert document.source == "paddleocr:gpu:0"
    assert not os.path.exists(str(state["image_path"]))


def test_paddle_cpu_forces_cpu_device(monkeypatch) -> None:
    state: dict[str, object] = {}

    class FakePaddleOCR:
        def __init__(self, **kwargs) -> None:
            state.update(kwargs)

    module = types.ModuleType("paddleocr")
    module.PaddleOCR = FakePaddleOCR  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "paddleocr", module)

    PaddleOcrEngine("paddle_cpu", Settings(paddle_device="gpu:0"))

    assert state["device"] == "cpu"


def test_factory_reuses_loaded_paddle_model(monkeypatch) -> None:
    state = {"initializations": 0}

    class FakePaddleOCR:
        def __init__(self, **_kwargs) -> None:
            state["initializations"] += 1

    module = types.ModuleType("paddleocr")
    module.PaddleOCR = FakePaddleOCR  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "paddleocr", module)
    settings = Settings(paddle_lang=None)

    clear_engine_cache()
    try:
        first = create_engine("paddle_cpu", settings)
        second = create_engine("paddle_cpu", settings)
    finally:
        clear_engine_cache()

    assert first is second
    assert state["initializations"] == 1


def test_multi_orientation_restores_boxes_to_original_image(monkeypatch) -> None:
    sizes: list[tuple[int, int]] = []

    class FakeResult:
        json = {
            "res": {
                "rec_texts": ["V542613"],
                "rec_scores": [0.99],
                "rec_boxes": [[0, 0, 10, 20]],
            }
        }

    class FakePaddleOCR:
        def __init__(self, **_kwargs) -> None:
            pass

        def predict(self, image_path: str):
            with Image.open(image_path) as image:
                sizes.append(image.size)
            return [FakeResult()]

    module = types.ModuleType("paddleocr")
    module.PaddleOCR = FakePaddleOCR  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "paddleocr", module)

    image_buffer = BytesIO()
    Image.new("RGB", (40, 20), "white").save(image_buffer, format="PNG")
    engine = PaddleOcrEngine("paddle_cpu", Settings())

    documents = engine.recognize_orientations(
        EngineInput(image_bytes=image_buffer.getvalue(), image_filename="sample.png")
    )

    assert sizes == [(40, 20), (20, 40), (20, 40)]
    assert documents["original"].items[0].box == (0.0, 0.0, 10.0, 20.0)
    assert documents["cw90"].items[0].box == (0.0, 10.0, 20.0, 20.0)
    assert documents["ccw90"].items[0].box == (20.0, 0.0, 40.0, 10.0)
    assert documents["cw90"].metadata["coordinates"] == "original_image"


def test_paddle_engine_recognizes_prepared_image_variants(monkeypatch) -> None:
    seen: list[str] = []

    class FakeResult:
        json = {
            "res": {
                "rec_texts": ["HASU", "5060396"],
                "rec_scores": [0.99, 0.98],
                "rec_boxes": [[1, 2, 3, 4], [5, 6, 7, 8]],
            }
        }

    class FakePaddleOCR:
        def __init__(self, **_kwargs) -> None:
            pass

        def predict(self, image_path: str):
            seen.append(Path(image_path).stem)
            return [FakeResult()]

    module = types.ModuleType("paddleocr")
    module.PaddleOCR = FakePaddleOCR  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "paddleocr", module)
    engine = PaddleOcrEngine("paddle_cpu", Settings())

    documents = engine.recognize_images(
        {
            "context_2x": Image.new("RGB", (100, 50), "white"),
            "grayscale_4x": Image.new("RGB", (200, 100), "gray"),
        },
        "sample.png",
    )

    assert seen == ["context_2x", "grayscale_4x"]
    assert [item.text for item in documents["context_2x"].items] == [
        "HASU",
        "5060396",
    ]
    assert documents["grayscale_4x"].metadata["orientation"] == "grayscale_4x"
