from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

try:
    from PIL import Image, ImageOps
except ImportError as exc:  # pragma: no cover - depends on the local environment
    raise SystemExit(
        "Pillow is required for the direct check-digit recognition experiment. "
        "Run this script in the DAS Photo AI Paddle image."
    ) from exc

from app.validators.iso6346 import append_check_digit, validate_container_number


VARIANTS = ("boxed_color", "boxed_gray", "inner_gray")
DEFAULT_MODEL_NAME = "PP-OCRv5_server_rec"
MIN_CANDIDATE_SCORE = 0.50
MIN_SINGLE_SCORE = 0.70


class TextRecognizer(Protocol):
    def recognize(self, image_path: Path) -> dict[str, Any]: ...


def _result_payload(result: Any) -> dict[str, Any]:
    payload = getattr(result, "json", None)
    if callable(payload):
        payload = payload()
    if payload is None and isinstance(result, dict):
        payload = result
    if not isinstance(payload, dict):
        raise RuntimeError("TextRecognition returned a result without a JSON dictionary")
    nested = payload.get("res")
    return nested if isinstance(nested, dict) else payload


class PaddleTextRecognizer:
    """Thin lazy wrapper around PaddleOCR's recognition-only module."""

    def __init__(self, model_name: str, device: str) -> None:
        try:
            from paddleocr import TextRecognition
        except ImportError as exc:
            raise RuntimeError(
                "PaddleOCR TextRecognition is unavailable. Run this script in the "
                "DAS Photo AI Paddle container."
            ) from exc
        try:
            self._model = TextRecognition(model_name=model_name, device=device)
        except Exception as exc:
            raise RuntimeError(
                f"TextRecognition failed to initialize on {device}: {exc}"
            ) from exc

    def recognize(self, image_path: Path) -> dict[str, Any]:
        started = time.perf_counter()
        try:
            results = list(self._model.predict(str(image_path), batch_size=1))
        except Exception as exc:
            raise RuntimeError(f"TextRecognition inference failed: {exc}") from exc
        if not results:
            raise RuntimeError("TextRecognition returned no result")
        payload = _result_payload(results[0])
        return {
            "text": str(payload.get("rec_text") or ""),
            "score": round(float(payload.get("rec_score") or 0.0), 6),
            "elapsed_ms": round((time.perf_counter() - started) * 1000, 3),
            "raw": payload,
        }


def normalize_single_digit(value: Any) -> str | None:
    compact = re.sub(r"[^A-Z0-9]", "", str(value or "").upper())
    return compact if len(compact) == 1 and compact.isdigit() else None


def _smooth(values: list[float], radius: int = 2) -> list[float]:
    output: list[float] = []
    for index in range(len(values)):
        start = max(0, index - radius)
        end = min(len(values), index + radius + 1)
        output.append(sum(values[start:end]) / (end - start))
    return output


def _strongest_pair(
    values: list[float],
    minimum_gap: int,
    maximum_gap: int,
    center_minimum: float,
    center_maximum: float,
) -> tuple[int, int] | None:
    ranked = sorted(
        range(len(values)), key=lambda index: values[index], reverse=True
    )[:40]
    best: tuple[float, int, int] | None = None
    for first in ranked:
        for second in ranked:
            if second <= first:
                continue
            gap = second - first
            center = (first + second) / 2
            if not minimum_gap <= gap <= maximum_gap:
                continue
            if not center_minimum <= center <= center_maximum:
                continue
            candidate = (values[first] + values[second], first, second)
            if best is None or candidate > best:
                best = candidate
    return None if best is None else (best[1], best[2])


def crop_inside_check_digit_box(
    image: Image.Image,
) -> tuple[Image.Image, dict[str, Any]]:
    """Remove the artificial crop padding and, when found, the printed box.

    The detector deliberately uses only pixel gradients. It does not know the
    expected check digit and cannot manufacture or choose a digit from ISO 6346.
    If a reliable rectangle cannot be located, the padded content is returned.
    """

    source = image.convert("RGB")
    source_width, source_height = source.size
    pad_x = int(source_width * 0.115)
    pad_y = int(source_height * 0.115)
    if source_width - pad_x * 2 < 3 or source_height - pad_y * 2 < 3:
        content = source
    else:
        content = source.crop(
            (pad_x, pad_y, source_width - pad_x, source_height - pad_y)
        )
    gray = ImageOps.autocontrast(ImageOps.grayscale(content))
    pixels = gray.load()
    width, height = gray.size
    if width < 8 or height < 8:
        return gray.convert("RGB"), {"box_detected": False}

    y_start, y_end = int(height * 0.08), max(1, int(height * 0.92))
    columns: list[float] = []
    for x in range(1, width - 1):
        differences = [
            abs(pixels[x + 1, y] - pixels[x - 1, y])
            for y in range(y_start, y_end)
        ]
        columns.append(
            float(sum(differences) + sum(value > 24 for value in differences) * 50)
        )
    columns = [0.0, *_smooth(columns), 0.0]
    vertical = _strongest_pair(
        columns,
        max(2, int(width * 0.18)),
        max(3, int(width * 0.50)),
        width * 0.25,
        width * 0.68,
    )
    if vertical is None:
        return gray.convert("RGB"), {"box_detected": False}
    left, right = vertical

    x_start = max(1, left - int(width * 0.03))
    x_end = min(width - 1, right + int(width * 0.03))
    rows: list[float] = []
    for y in range(1, height - 1):
        differences = [
            abs(pixels[x, y + 1] - pixels[x, y - 1])
            for x in range(x_start, x_end)
        ]
        rows.append(
            float(sum(differences) + sum(value > 24 for value in differences) * 50)
        )
    rows = [0.0, *_smooth(rows), 0.0]
    horizontal = _strongest_pair(
        rows,
        max(2, int(height * 0.35)),
        max(3, int(height * 0.90)),
        height * 0.30,
        height * 0.70,
    )
    if horizontal is None:
        return gray.convert("RGB"), {"box_detected": False}
    top, bottom = horizontal

    inset_x = max(2, int((right - left) * 0.06))
    inset_y = max(2, int((bottom - top) * 0.05))
    inner_box = (
        min(right - 1, left + inset_x),
        min(bottom - 1, top + inset_y),
        max(left + 1, right - inset_x),
        max(top + 1, bottom - inset_y),
    )
    if inner_box[2] <= inner_box[0] or inner_box[3] <= inner_box[1]:
        return gray.convert("RGB"), {"box_detected": False}

    inner = ImageOps.autocontrast(ImageOps.grayscale(content.crop(inner_box)))
    corners = (
        (0, 0),
        (width - 1, 0),
        (0, height - 1),
        (width - 1, height - 1),
    )
    background = int(sum(gray.getpixel(point) for point in corners) / len(corners))
    border_x = max(8, int(inner.width * 0.35))
    border_y = max(8, int(inner.height * 0.20))
    inner = ImageOps.expand(
        inner,
        border=(border_x, border_y, border_x, border_y),
        fill=background,
    )
    return inner.convert("RGB"), {
        "box_detected": True,
        "detected_box": [left, top, right, bottom],
        "inner_box": list(inner_box),
    }


def create_recognition_variants(
    source: Path, output_root: Path, file_stem: str
) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    with Image.open(source) as opened:
        color = ImageOps.exif_transpose(opened).convert("RGB")
        images: dict[str, tuple[Image.Image, dict[str, Any]]] = {
            "boxed_color": (color.copy(), {"box_detected": None}),
            "boxed_gray": (
                ImageOps.autocontrast(ImageOps.grayscale(color)).convert("RGB"),
                {"box_detected": None},
            ),
        }
        images["inner_gray"] = crop_inside_check_digit_box(color)

        for variant, (image, metadata) in images.items():
            directory = output_root / variant
            directory.mkdir(parents=True, exist_ok=True)
            destination = directory / f"{file_stem}__{variant}.png"
            image.save(destination, format="PNG")
            output[variant] = {
                "path": destination,
                "preprocessing": metadata,
            }
    return output


def select_digit_without_truth(
    recognitions: dict[str, dict[str, Any]],
    *,
    minimum_candidate_score: float = MIN_CANDIDATE_SCORE,
    minimum_single_score: float = MIN_SINGLE_SCORE,
) -> dict[str, Any]:
    """Fuse recognition results without consulting ISO or the answer sheet."""

    eligible: list[tuple[str, str, float]] = []
    for variant, result in recognitions.items():
        digit = normalize_single_digit(result.get("text"))
        score = float(result.get("score") or 0.0)
        if digit is not None and score >= minimum_candidate_score:
            eligible.append((variant, digit, score))

    if not eligible:
        return {
            "status": "not_found",
            "digit": None,
            "reason": "no_single_digit_above_threshold",
            "supporting_variants": [],
            "score": 0.0,
        }

    by_digit: dict[str, list[tuple[str, float]]] = defaultdict(list)
    for variant, digit, score in eligible:
        by_digit[digit].append((variant, score))
    ranking = sorted(
        by_digit.items(),
        key=lambda item: (len(item[1]), max(score for _variant, score in item[1])),
        reverse=True,
    )
    best_digit, best_support = ranking[0]
    best_count = len(best_support)
    tied = len(ranking) > 1 and len(ranking[1][1]) == best_count
    best_score = max(score for _variant, score in best_support)

    if best_count >= 2 and not tied:
        return {
            "status": "selected",
            "digit": best_digit,
            "reason": "variant_consensus",
            "supporting_variants": [variant for variant, _score in best_support],
            "score": round(best_score, 6),
        }
    if len(by_digit) == 1 and best_score >= minimum_single_score:
        return {
            "status": "selected",
            "digit": best_digit,
            "reason": "high_confidence_single_result",
            "supporting_variants": [variant for variant, _score in best_support],
            "score": round(best_score, 6),
        }
    return {
        "status": "conflict" if len(by_digit) > 1 else "low_confidence",
        "digit": None,
        "reason": (
            "different_digits_between_variants"
            if len(by_digit) > 1
            else "single_result_below_auto_select_threshold"
        ),
        "supporting_variants": [],
        "score": round(best_score, 6),
        "candidates": {
            digit: [
                {"variant": variant, "score": round(score, 6)}
                for variant, score in support
            ]
            for digit, support in sorted(by_digit.items())
        },
    }


def _load_previous_run(previous_run_dir: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    details_path = previous_run_dir / "details.json"
    summary_path = previous_run_dir / "summary.json"
    if not details_path.is_file():
        raise SystemExit(f"previous details not found: {details_path}")
    if not summary_path.is_file():
        raise SystemExit(f"previous summary not found: {summary_path}")
    details = json.loads(details_path.read_text(encoding="utf-8"))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if not isinstance(details, list) or not isinstance(summary, dict):
        raise SystemExit("previous run has an invalid details.json or summary.json")
    return details, summary


def _prefix_from_previous_record(record: dict[str, Any]) -> str | None:
    original = (record.get("variants") or {}).get("original") or {}
    observed = re.sub(r"[^A-Z0-9]", "", str(original.get("observed") or "").upper())
    return observed if len(observed) == 10 else None


def summarize_records(
    records: list[dict[str, Any]], previous_summary: dict[str, Any]
) -> dict[str, Any]:
    previous_metrics = previous_summary.get("metrics") or {}
    by_variant: dict[str, Any] = {}
    for variant in VARIANTS:
        rows = [
            (record.get("recognitions") or {}).get(variant)
            for record in records
            if (record.get("recognitions") or {}).get(variant)
        ]
        by_variant[variant] = {
            "tested": len(rows),
            "single_digit_detected": sum(bool(row.get("digit")) for row in rows),
            "digit_exact": sum(bool(row.get("digit_exact")) for row in rows),
            "average_score": (
                round(sum(float(row.get("score") or 0.0) for row in rows) / len(rows), 6)
                if rows
                else 0.0
            ),
        }

    rescued = sum(bool(record.get("selected_verified_exact")) for record in records)
    false_verified = sum(
        record.get("verification") == "verified"
        and not record.get("selected_verified_exact")
        for record in records
    )
    inherited_verified = int(previous_metrics.get("selected_verified_exact") or 0)
    previous_evaluated = int(previous_metrics.get("evaluated") or len(records))
    return {
        "evaluated_previous_failures": len(records),
        "direct_recognition_rescued": rescued,
        "direct_recognition_accuracy": round(rescued / len(records), 4) if records else 0.0,
        "selected_false_verified": false_verified,
        "selection_statuses": dict(
            sorted(Counter(str(record.get("selection_status")) for record in records).items())
        ),
        "selection_reasons": dict(
            sorted(Counter(str(record.get("selection_reason")) for record in records).items())
        ),
        "by_variant": by_variant,
        "previous_run_scope": {
            "evaluated": previous_evaluated,
            "already_verified": inherited_verified,
            "combined_verified": inherited_verified + rescued,
            "combined_accuracy": (
                round((inherited_verified + rescued) / previous_evaluated, 4)
                if previous_evaluated
                else 0.0
            ),
        },
        "still_unverified_or_wrong": [
            record.get("file_name")
            for record in records
            if not record.get("selected_verified_exact")
        ],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run PaddleOCR TextRecognition directly on v0.3.2 check-digit crops, "
            "bypassing text detection and testing box-removal preprocessing"
        )
    )
    parser.add_argument("--previous-run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--device", default="gpu:0")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--minimum-candidate-score", type=float, default=MIN_CANDIDATE_SCORE)
    parser.add_argument("--minimum-single-score", type=float, default=MIN_SINGLE_SCORE)
    return parser


def main(recognizer: TextRecognizer | None = None) -> int:
    args = build_parser().parse_args()
    details, previous_summary = _load_previous_run(args.previous_run_dir)
    previous_failures = set(
        str(name)
        for name in ((previous_summary.get("metrics") or {}).get("still_unverified_or_wrong") or [])
    )
    rows = [record for record in details if str(record.get("file_name")) in previous_failures]
    if args.limit > 0:
        rows = rows[: args.limit]
    if not rows:
        raise SystemExit("previous run contains no unresolved files")

    crop_root = args.previous_run_dir / "crops" / "check_digit_strip_4x"
    if not crop_root.is_dir():
        raise SystemExit(f"v0.3.2 check-digit crops not found: {crop_root}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    details_path = args.output_dir / "details.ndjson"
    if details_path.exists():
        raise SystemExit(f"output already contains details.ndjson: {args.output_dir}")

    active_recognizer = recognizer or PaddleTextRecognizer(args.model_name, args.device)
    variants_root = args.output_dir / "crops"
    records: list[dict[str, Any]] = []
    started = time.perf_counter()
    with details_path.open("a", encoding="utf-8") as detail_file:
        for index, previous_record in enumerate(rows, start=1):
            file_name = str(previous_record.get("file_name") or "")
            file_stem = str(previous_record.get("file_stem") or Path(file_name).stem)
            expected = re.sub(
                r"[^A-Z0-9]", "", str(previous_record.get("expected") or "").upper()
            )
            prefix = _prefix_from_previous_record(previous_record)
            source = crop_root / f"{file_stem}__check_digit_strip_4x.png"
            record: dict[str, Any] = {
                "file_name": file_name,
                "file_stem": file_stem,
                "expected": expected,
                "prefix_observed": prefix,
                "source_crop": str(source),
                "recognitions": {},
                "tested_at": datetime.now(timezone.utc).isoformat(),
            }
            try:
                if not source.is_file():
                    raise FileNotFoundError(f"check-digit crop not found: {source}")
                if prefix is None:
                    raise ValueError("previous original result does not contain a 10-character prefix")
                variant_files = create_recognition_variants(source, variants_root, file_stem)
                for variant in VARIANTS:
                    result = active_recognizer.recognize(variant_files[variant]["path"])
                    digit = normalize_single_digit(result.get("text"))
                    record["recognitions"][variant] = {
                        **result,
                        "raw": result.get("raw") or {},
                        "digit": digit,
                        "digit_exact": bool(expected and digit == expected[-1]),
                        "preprocessing": variant_files[variant]["preprocessing"],
                    }

                selection = select_digit_without_truth(
                    record["recognitions"],
                    minimum_candidate_score=args.minimum_candidate_score,
                    minimum_single_score=args.minimum_single_score,
                )
                selected_digit = selection.get("digit")
                observed = f"{prefix}{selected_digit}" if selected_digit is not None else prefix
                verification = (
                    "verified"
                    if selected_digit is not None and validate_container_number(observed)
                    else "mismatch" if selected_digit is not None else "unverified"
                )
                record.update(
                    {
                        "selection_status": selection["status"],
                        "selection_reason": selection["reason"],
                        "selection": selection,
                        "selected_digit": selected_digit,
                        "selected_observed": observed,
                        "calculated_value": append_check_digit(prefix),
                        "verification": verification,
                        "selected_verified_exact": bool(
                            verification == "verified" and observed == expected
                        ),
                    }
                )
            except Exception as exc:
                record.update(
                    {
                        "selection_status": "error",
                        "selection_reason": "processing_error",
                        "error": f"{type(exc).__name__}: {exc}",
                        "verification": "unverified",
                        "selected_verified_exact": False,
                    }
                )
            records.append(record)
            detail_file.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
            detail_file.flush()
            print(
                f"[{index}/{len(rows)}] {file_name}: "
                f"{record.get('selection_status')} "
                f"{record.get('selected_observed') or '-'} "
                f"{record.get('verification')}",
                flush=True,
            )

    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "wall_elapsed_seconds": round(time.perf_counter() - started, 3),
        "experiment": "direct_check_digit_text_recognition",
        "model_name": args.model_name,
        "device": args.device,
        "previous_run_dir": str(args.previous_run_dir),
        "variants": list(VARIANTS),
        "thresholds": {
            "minimum_candidate_score": args.minimum_candidate_score,
            "minimum_single_score": args.minimum_single_score,
        },
        "selection_policy": (
            "Select only an OCR-observed single digit by cross-variant consensus or "
            "one high-confidence result; use ISO 6346 only after selection to validate."
        ),
        "metrics": summarize_records(records, previous_summary),
    }
    (args.output_dir / "details.json").write_text(
        json.dumps(records, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"output={args.output_dir.resolve()}")
    return 1 if any(record.get("selection_status") == "error" for record in records) else 0


if __name__ == "__main__":
    sys.exit(main())
