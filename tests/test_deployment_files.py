from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_compose_uses_verified_gpu_image_and_persistent_caches() -> None:
    compose = yaml.safe_load((PROJECT_ROOT / "compose.yaml").read_text(encoding="utf-8"))
    service = compose["services"]["das-photo-ai"]

    assert service["runtime"] == "nvidia"
    assert service["gpus"] == "all"
    assert service["environment"]["DAS_AI_ENGINE"] == "paddle_gpu"
    assert service["environment"]["DAS_AI_PRELOAD_MODEL"] == "1"
    assert any("/root/.paddlex" in volume for volume in service["volumes"])
    assert any("/root/.paddleocr" in volume for volume in service["volumes"])


def test_dockerfile_uses_cuda_12_9_paddle_base() -> None:
    dockerfile = (PROJECT_ROOT / "Dockerfile").read_text(encoding="utf-8")

    assert "paddlepaddle/paddle:3.3.0-gpu-cuda12.9-cudnn9.9" in dockerfile
    assert "requirements-paddle.txt" in dockerfile
