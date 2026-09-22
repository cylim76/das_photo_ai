from fastapi.testclient import TestClient

from app.engines.base import EngineUnavailableError
from app.main import app


client = TestClient(app)


def test_health() -> None:
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_models_exposes_stage_one_engines() -> None:
    response = client.get("/api/v1/models")
    assert response.status_code == 200
    names = {engine["name"] for engine in response.json()["engines"]}
    assert names == {"mock", "json", "paddle_cpu", "paddle_gpu"}
    assert response.json()["paddleocr"]["available"] is False


def test_mock_recognition() -> None:
    response = client.post(
        "/api/v1/recognize",
        json={
            "engine": "mock",
            "targets": ["container_number", "seal_number"],
            "mock_profile": "default",
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["engine"] == "mock"
    assert payload["results"]["container_number"]["status"] == "found"
    assert payload["results"]["container_number"]["verification"] == "unverified"
    assert payload["results"]["seal_number"]["value"] == "V542613"


def test_json_recognition_requires_payload() -> None:
    response = client.post(
        "/api/v1/recognize",
        json={"engine": "json", "targets": ["container_number"]},
    )
    assert response.status_code == 422
    assert response.json()["error"] == "engine_input_error"


def test_json_recognition_accepts_paddle_result() -> None:
    response = client.post(
        "/api/v1/recognize",
        json={
            "engine": "json",
            "targets": ["container_number"],
            "ocr_result": {
                "rec_texts": ["HASU", "506039"],
                "rec_scores": [0.99, 0.998],
                "rec_boxes": [[100, 100, 200, 145], [215, 100, 360, 145]],
            },
        },
    )
    assert response.status_code == 200
    result = response.json()["results"]["container_number"]
    assert result["status"] == "found"
    assert result["validation"] == "inferred"
    assert result["verification"] == "unverified"
    assert result["value"] == "HASU506039"
    assert result["observed_value"] == "HASU506039"
    assert result["suggested_value"] == "HASU5060396"
    assert result["observed_check_digit"] is None
    assert result["calculated_check_digit"] == "6"


def test_image_endpoint_supports_mock_engine_without_paddle() -> None:
    response = client.post(
        "/api/v1/recognize/image",
        data={"engine": "mock", "targets": "container_number,seal_number"},
        files={"image": ("sample.jpg", b"fake-image-bytes", "image/jpeg")},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["engine"] == "mock"
    assert payload["results"]["container_number"]["status"] == "found"


def test_image_endpoint_rejects_json_engine() -> None:
    response = client.post(
        "/api/v1/recognize/image",
        data={"engine": "json"},
        files={"image": ("sample.jpg", b"fake-image-bytes", "image/jpeg")},
    )

    assert response.status_code == 422


def test_image_endpoint_rejects_non_image_content_type() -> None:
    response = client.post(
        "/api/v1/recognize/image",
        data={"engine": "mock"},
        files={"image": ("sample.txt", b"not-an-image", "text/plain")},
    )

    assert response.status_code == 415


def test_image_endpoint_reports_unavailable_paddle_as_503(monkeypatch) -> None:
    def unavailable(*_args, **_kwargs):
        raise EngineUnavailableError("PaddleOCR unavailable for test")

    monkeypatch.setattr("app.service.create_engine", unavailable)
    response = client.post(
        "/api/v1/recognize/image",
        data={"engine": "paddle_gpu"},
        files={"image": ("sample.jpg", b"fake-image-bytes", "image/jpeg")},
    )

    assert response.status_code == 503
    assert response.json()["error"] == "engine_unavailable"
