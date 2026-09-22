from app.domain import OCRDocument, OCRItem
from app.pipelines.seal_number import extract_seal_numbers


def test_extracts_seal_and_filters_date() -> None:
    document = OCRDocument(
        source="test",
        items=(
            OCRItem("V542613", 0.999, source_index=0),
            OCRItem("2023.10.27 14:05", 0.98, source_index=1),
        ),
    )

    candidates = extract_seal_numbers(document)

    assert candidates[0].value == "V542613"
    assert all("20231027" not in candidate.value for candidate in candidates)


def test_filters_container_number_and_weight_labels() -> None:
    document = OCRDocument(
        source="test",
        items=(
            OCRItem("MSCU6639870", 0.99, source_index=0),
            OCRItem("MAX.GROSS", 0.99, source_index=1),
            OCRItem("32.500 kg", 0.99, source_index=2),
        ),
    )

    candidates = extract_seal_numbers(document)

    assert all(candidate.value != "MSCU6639870" for candidate in candidates)
    assert all("MAXGROSS" not in candidate.value for candidate in candidates)

