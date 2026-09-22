from __future__ import annotations

import re


LETTER_VALUES = {
    "A": 10,
    "B": 12,
    "C": 13,
    "D": 14,
    "E": 15,
    "F": 16,
    "G": 17,
    "H": 18,
    "I": 19,
    "J": 20,
    "K": 21,
    "L": 23,
    "M": 24,
    "N": 25,
    "O": 26,
    "P": 27,
    "Q": 28,
    "R": 29,
    "S": 30,
    "T": 31,
    "U": 32,
    "V": 34,
    "W": 35,
    "X": 36,
    "Y": 37,
    "Z": 38,
}

PREFIX_PATTERN = re.compile(r"^[A-Z]{3}[UJZ][0-9]{6}$")
FULL_PATTERN = re.compile(r"^[A-Z]{3}[UJZ][0-9]{7}$")


def calculate_check_digit(prefix: str) -> int:
    normalized = re.sub(r"[^A-Z0-9]", "", prefix.upper())
    if not PREFIX_PATTERN.fullmatch(normalized):
        raise ValueError("ISO 6346 prefix must be three letters, U/J/Z, and six digits")

    total = 0
    for position, character in enumerate(normalized):
        value = int(character) if character.isdigit() else LETTER_VALUES[character]
        total += value * (2**position)
    remainder = total % 11
    return 0 if remainder == 10 else remainder


def append_check_digit(prefix: str) -> str:
    normalized = re.sub(r"[^A-Z0-9]", "", prefix.upper())
    return f"{normalized}{calculate_check_digit(normalized)}"


def validate_container_number(value: str) -> bool:
    normalized = re.sub(r"[^A-Z0-9]", "", value.upper())
    if not FULL_PATTERN.fullmatch(normalized):
        return False
    return calculate_check_digit(normalized[:10]) == int(normalized[10])

