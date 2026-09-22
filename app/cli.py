from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.config import get_settings
from app.schemas.recognition import RecognizeRequest
from app.service import recognize


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Extract DAS fields from PaddleOCR JSON")
    parser.add_argument("json_file", type=Path, help="PaddleOCR *_res.json file")
    parser.add_argument(
        "--target",
        choices=("container_number", "seal_number"),
        action="append",
        dest="targets",
        help="Target field. May be supplied more than once.",
    )
    parser.add_argument("--compact", action="store_true", help="Print compact JSON")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        payload = json.loads(args.json_file.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise SystemExit(f"File not found: {args.json_file}") from exc
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Invalid JSON: {exc}") from exc

    targets = args.targets or ["container_number", "seal_number"]
    response = recognize(
        RecognizeRequest(engine="json", targets=targets, ocr_result=payload),
        get_settings(),
    )
    print(response.model_dump_json(indent=None if args.compact else 2))


if __name__ == "__main__":
    main()

