from app.domain import OCRDocument, OCRItem
from app.pipelines.container_number import extract_container_numbers
from app.validators.iso6346 import append_check_digit


def test_combines_split_container_number_and_infers_check_digit() -> None:
    document = OCRDocument(
        source="test",
        items=(
            OCRItem("HASU", 0.9904, box=(100, 100, 200, 145), source_index=0),
            OCRItem("506039", 0.9976, box=(215, 100, 360, 145), source_index=1),
        ),
    )

    candidates = extract_container_numbers(document)

    assert candidates
    assert candidates[0].observed_value == "HASU506039"
    assert candidates[0].value == "HASU506039"
    assert candidates[0].suggested_value == append_check_digit("HASU506039")
    assert candidates[0].validation == "inferred"
    assert candidates[0].verification == "unverified"
    assert candidates[0].observed_check_digit is None
    assert candidates[0].calculated_check_digit == candidates[0].suggested_value[-1]
    assert candidates[0].check_digit_source == "calculated"
    assert candidates[0].source_indices == (0, 1)


def test_prefers_complete_valid_number() -> None:
    document = OCRDocument(
        source="test",
        items=(OCRItem("MSCU6639870", 0.99, source_index=0),),
    )

    candidates = extract_container_numbers(document)

    assert candidates[0].value == "MSCU6639870"
    assert candidates[0].validation == "valid"
    assert candidates[0].verification == "verified"
    assert candidates[0].observed_check_digit == "0"
    assert candidates[0].calculated_check_digit == "0"
    assert candidates[0].check_digit_source == "observed"
    assert candidates[0].inferred is False


def test_reports_observed_check_digit_mismatch_without_hiding_it() -> None:
    document = OCRDocument(
        source="test",
        items=(OCRItem("MSCU6639871", 0.99, source_index=0),),
    )

    candidates = extract_container_numbers(document)
    mismatch = candidates[0]

    assert mismatch.value == "MSCU6639871"
    assert mismatch.suggested_value == "MSCU6639870"
    assert mismatch.observed_check_digit == "1"
    assert mismatch.calculated_check_digit == "0"
    assert mismatch.verification == "mismatch"


def test_corrects_common_digit_letter_confusion() -> None:
    document = OCRDocument(
        source="test",
        items=(OCRItem("MSCU6639B70", 0.97, source_index=0),),
    )

    candidates = extract_container_numbers(document)

    assert any(candidate.value == "MSCU6639870" for candidate in candidates)
    corrected = next(candidate for candidate in candidates if candidate.value == "MSCU6639870")
    assert corrected.corrections == 1
    assert corrected.validation == "valid"
