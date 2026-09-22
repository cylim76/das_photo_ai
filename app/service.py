from __future__ import annotations

import logging
import time
import uuid

from app.config import Settings
from app.domain import EngineInput, OCRDocument
from app.engines import create_engine
from app.engines.check_digit_recognizer import create_check_digit_recognizer
from app.pipelines import extract_container_numbers, extract_seal_numbers
from app.pipelines.check_digit_fallback import apply_check_digit_fallback
from app.pipelines.common import ExtractionCandidate
from app.pipelines.seal_fusion import fuse_seal_candidates
from app.schemas.recognition import (
    CandidateResponse,
    FieldResultResponse,
    OCRItemResponse,
    RecognizeRequest,
    RecognizeResponse,
    TargetName,
)


EXTRACTOR_VERSION = "0.4.0"
logger = logging.getLogger(__name__)


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
        orientation=candidate.orientation,
        supporting_orientations=list(candidate.supporting_orientations),
        selection_reasons=list(candidate.selection_reasons),
    )


def _field_result(
    target: TargetName,
    candidates: list[ExtractionCandidate],
    postprocessing: dict[str, object] | None = None,
) -> FieldResultResponse:
    responses = [_candidate_response(candidate) for candidate in candidates]
    if not responses:
        return FieldResultResponse(
            target=target,
            status="not_found",
            postprocessing=postprocessing or {},
        )
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
        orientation=best.orientation,
        candidates=responses,
        postprocessing=postprocessing or {},
    )


def _apply_container_check_digit_fallback(
    *,
    candidates: list[ExtractionCandidate],
    image_bytes: bytes | None,
    engine_name: str,
    settings: Settings,
) -> tuple[list[ExtractionCandidate], dict[str, object]]:
    if (
        not settings.container_check_digit_fallback
        or not image_bytes
        or not engine_name.startswith("paddle_")
    ):
        return candidates, {}
    if not candidates:
        return candidates, {"status": "not_eligible", "reason": "no_container_candidate"}

    best = candidates[0]
    if (
        best.verification != "unverified"
        or len(best.observed_value) != 10
        or not best.boxes
    ):
        return candidates, {
            "status": "not_needed",
            "reason": "best_candidate_is_not_an_unverified_10_character_value",
        }

    device = "cpu" if engine_name == "paddle_cpu" else settings.paddle_device
    try:
        recognizer = create_check_digit_recognizer(
            settings.check_digit_model,
            device,
        )
        completed, metadata = apply_check_digit_fallback(
            image_bytes,
            best,
            recognizer,
            minimum_candidate_score=settings.check_digit_min_candidate_score,
            minimum_single_score=settings.check_digit_min_single_score,
        )
        if completed is None:
            return candidates, metadata
        remaining = [candidate for candidate in candidates[1:] if candidate.value != completed.value]
        return [completed, *remaining], metadata
    except Exception as exc:  # Optional fallback must not erase the primary OCR result.
        logger.warning("container check-digit fallback failed: %s", exc)
        return candidates, {
            "status": "error",
            "strategy": "english_check_digit_direct_recognition",
            "model_name": settings.check_digit_model,
            "error": f"{type(exc).__name__}: {exc}",
        }


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
    recognize_orientations = getattr(engine, "recognize_orientations", None)
    if (
        input_data.image_bytes
        and "seal_number" in targets
        and settings.seal_multi_orientation
        and callable(recognize_orientations)
    ):
        documents = recognize_orientations(input_data)
        document = documents["original"]
    else:
        document = engine.recognize(input_data)
        documents = {"original": document}

    results: dict[TargetName, FieldResultResponse] = {}
    for target in targets:
        postprocessing: dict[str, object] = {}
        if target == "container_number":
            candidates = extract_container_numbers(document, settings.max_candidates)
            candidates, postprocessing = _apply_container_check_digit_fallback(
                candidates=candidates,
                image_bytes=input_data.image_bytes,
                engine_name=engine.name,
                settings=settings,
            )
        else:
            candidates_by_orientation = {
                orientation: extract_seal_numbers(variant, settings.max_candidates)
                for orientation, variant in documents.items()
            }
            candidates = fuse_seal_candidates(
                candidates_by_orientation,
                settings.max_candidates,
            )
        results[target] = _field_result(target, candidates, postprocessing)

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
