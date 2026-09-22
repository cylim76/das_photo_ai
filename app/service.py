from __future__ import annotations

import time
import uuid

from app.config import Settings
from app.domain import EngineInput, OCRDocument
from app.engines import create_engine
from app.pipelines import extract_container_numbers, extract_seal_numbers
from app.pipelines.common import ExtractionCandidate
from app.schemas.recognition import (
    CandidateResponse,
    FieldResultResponse,
    OCRItemResponse,
    RecognizeRequest,
    RecognizeResponse,
    TargetName,
)


EXTRACTOR_VERSION = "0.2.0"


def _candidate_response(candidate: ExtractionCandidate) -> CandidateResponse:
    return CandidateResponse(
        value=candidate.value,
        observed_value=candidate.observed_value,
        suggested_value=candidate.suggested_value,
        ocr_confidence=candidate.ocr_confidence,
        extraction_score=candidate.extraction_score,
        validation=candidate.validation,
        verification=candidate.verification,
        observed_check_digit=candidate.observed_check_digit,
        calculated_check_digit=candidate.calculated_check_digit,
        check_digit_source=candidate.check_digit_source,
        inferred=candidate.inferred,
        corrections=candidate.corrections,
        source_texts=list(candidate.source_texts),
        source_indices=list(candidate.source_indices),
        boxes=[list(box) for box in candidate.boxes],
    )


def _field_result(
    target: TargetName,
    candidates: list[ExtractionCandidate],
) -> FieldResultResponse:
    responses = [_candidate_response(candidate) for candidate in candidates]
    if not responses:
        return FieldResultResponse(target=target, status="not_found")
    best = responses[0]
    return FieldResultResponse(
        target=target,
        status="found",
        value=best.value,
        observed_value=best.observed_value,
        suggested_value=best.suggested_value,
        ocr_confidence=best.ocr_confidence,
        extraction_score=best.extraction_score,
        validation=best.validation,
        verification=best.verification,
        observed_check_digit=best.observed_check_digit,
        calculated_check_digit=best.calculated_check_digit,
        check_digit_source=best.check_digit_source,
        candidates=responses,
    )


def _ocr_items(document: OCRDocument) -> list[OCRItemResponse]:
    return [
        OCRItemResponse(
            text=item.text,
            score=item.score,
            polygon=[list(point) for point in item.polygon],
            box=list(item.box) if item.box is not None else None,
            source_index=item.source_index,
        )
        for item in document.items
    ]


def _recognize_with_input(
    *,
    engine_name: str,
    targets: list[TargetName],
    input_data: EngineInput,
    request_id: str | None,
    settings: Settings,
) -> RecognizeResponse:
    started = time.perf_counter()
    engine = create_engine(engine_name, settings)
    document = engine.recognize(input_data)

    results: dict[TargetName, FieldResultResponse] = {}
    for target in targets:
        if target == "container_number":
            candidates = extract_container_numbers(document, settings.max_candidates)
        else:
            candidates = extract_seal_numbers(document, settings.max_candidates)
        results[target] = _field_result(target, candidates)

    elapsed_ms = round((time.perf_counter() - started) * 1000, 3)
    return RecognizeResponse(
        request_id=request_id or uuid.uuid4().hex,
        service_version=settings.version,
        extractor_version=EXTRACTOR_VERSION,
        engine=engine.name,
        elapsed_ms=elapsed_ms,
        ocr_source=document.source,
        ocr_items=_ocr_items(document),
        results=results,
    )


def recognize(request: RecognizeRequest, settings: Settings) -> RecognizeResponse:
    return _recognize_with_input(
        engine_name=request.engine or settings.default_engine,
        targets=request.targets,
        input_data=EngineInput(
            ocr_payload=request.ocr_result,
            mock_profile=request.mock_profile,
        ),
        request_id=request.request_id,
        settings=settings,
    )


def recognize_image(
    *,
    image_bytes: bytes,
    image_filename: str | None,
    targets: list[TargetName],
    engine_name: str,
    request_id: str | None,
    settings: Settings,
) -> RecognizeResponse:
    return _recognize_with_input(
        engine_name=engine_name,
        targets=targets,
        input_data=EngineInput(
            image_bytes=image_bytes,
            image_filename=image_filename,
        ),
        request_id=request_id,
        settings=settings,
    )
