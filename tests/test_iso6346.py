import pytest

from app.validators.iso6346 import (
    append_check_digit,
    calculate_check_digit,
    validate_container_number,
)


def test_known_iso_6346_example_is_valid() -> None:
    assert calculate_check_digit("MSCU663987") == 0
    assert append_check_digit("MSCU663987") == "MSCU6639870"
    assert validate_container_number("MSCU 663987-0") is True


def test_invalid_check_digit_is_rejected() -> None:
    assert validate_container_number("MSCU6639871") is False


def test_invalid_prefix_raises() -> None:
    with pytest.raises(ValueError):
        calculate_check_digit("NOT-A-CONTAINER")

