from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


TargetName = Literal["container_number", "seal_number"]
EngineName = Literal["mock", "json", "paddle_cpu", "paddle_gpu"]


class RecognizeRequest(BaseModel):
    engine: EngineName | None = Field(
        default=None,
        description="OCR engine. Uses DAS_AI_ENGINE when omitted.",
    )
    targets: list[TargetName] = Field(
        default_factory=lambda: ["container_number", "seal_number"]
    )
    ocr_result: dict[str, Any] | None = Field(
        default=None,
        description="Raw PaddleOCR JSON or normalized OCR document for the json engine.",
    )
    mock_profile: str = "default"
    request_id: str | None = Field(default=None, max_length=100)

    @field_validator("targets")
    @classmethod
    def validate_targets(cls, value: list[TargetName]) -> list[TargetName]:
        if not value:
            raise ValueError("targets must contain at least one target")
        return list(dict.fromkeys(value))


class OCRItemResponse(BaseModel):
    text: str
    score: float
    polygon: list[list[float]] = Field(default_factory=list)
    box: list[float] | None = None
    source_index: int


class CandidateResponse(BaseModel):
    value: str
    observed_value: str
    suggested_value: str | None = None
    ocr_confidence: float
    extraction_score: float
    validation: str | None = None
    verification: Literal["verified", "unverified", "mismatch", "not_applicable"] = (
        "not_applicable"
    )
    observed_check_digit: str | None = None
    calculated_check_digit: str | None = None
    check_digit_source: Literal["observed", "calculated"] | None = None
    inferred: bool = False
    corrections: int = 0
    source_texts: list[str] = Field(default_factory=list)
    source_indices: list[int] = Field(default_factory=list)
    boxes: list[list[float]] = Field(default_factory=list)
    orientation: str = "original"
    supporting_orientations: list[str] = Field(default_factory=list)
    selection_reasons: list[str] = Field(default_factory=list)


class FieldResultResponse(BaseModel):
    target: TargetName
    status: Literal["found", "not_found"]
    value: str | None = None
    observed_value: str | None = None
    suggested_value: str | None = None
    ocr_confidence: float | None = None
    extraction_score: float | None = None
    validation: str | None = None
    verification: Literal["verified", "unverified", "mismatch", "not_applicable"] | None = None
    observed_check_digit: str | None = None
    calculated_check_digit: str | None = None
    check_digit_source: Literal["observed", "calculated"] | None = None
    orientation: str | None = None
    candidates: list[CandidateResponse] = Field(default_factory=list)
    postprocessing: dict[str, Any] = Field(default_factory=dict)


class RecognizeResponse(BaseModel):
    request_id: str
    service_version: str
    extractor_version: str
    engine: str
    elapsed_ms: float
    ocr_source: str
    ocr_items: list[OCRItemResponse]
    results: dict[TargetName, FieldResultResponse]
