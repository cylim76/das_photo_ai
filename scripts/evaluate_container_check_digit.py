from __future__ import annotations

import argparse
import copy
import json
import math
import re
import shutil
import sys
import tempfile
import time
import urllib.error
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from PIL import Image, ImageOps
except ImportError as exc:  # pragma: no cover - depends on the local environment
    raise SystemExit(
        "Pillow is required for the container check-digit experiment. "
        "Install requirements-dev.txt or run this script in the DAS Photo AI image."
    ) from exc

from app.validators.iso6346 import append_check_digit, validate_container_number
from scripts.evaluate_image_api import (
    evaluate_payload,
    get_json,
    load_report_rows,
    post_image,
)


LEGACY_VARIANTS = (
    "original",
    "context_2x",
    "context_4x",
    "grayscale_4x",
    "right_30_2x",
    "right_30_4x",
    "right_30_grayscale_4x",
)
V032_VARIANTS = (
    "original",
    "right_30_vpad15_2x",
    "right_30_vpad15_deskew_2x",
    "check_digit_strip_4x",
)
PROFILE_VARIANTS = {
    "legacy": LEGACY_VARIANTS,
    "v032": V032_VARIANTS,
}
VARIANTS = tuple(dict.fromkeys((*LEGACY_VARIANTS, *V032_VARIANTS)))
MIN_ISOLATED_DIGIT_SCORE = 0.50


def _container_result(payload: dict[str, Any]) -> dict[str, Any]:
    return (payload.get("results") or {}).get("container_number") or {}


def _ten_character_candidate(payload: dict[str, Any]) -> dict[str, Any] | None:
    result = _container_result(payload)
    return next(
        (
            item
            for item in result.get("candidates") or []
            if len(str(item.get("observed_value") or "")) == 10
            and item.get("boxes")
        ),
        None,
    )


def _candidate_region(payload: dict[str, Any]) -> tuple[float, float, float, float] | None:
    """Return the box of the ten observed ISO characters, not the suggestion.

    The calculated check digit is intentionally ignored. The crop only uses the
    location of OCR evidence already present in the original image.
    """

    candidate = _ten_character_candidate(payload)
    if candidate is None:
        return None
    boxes = [box for box in candidate.get("boxes") or [] if len(box) == 4]
    if not boxes:
        return None
    return (
        min(float(box[0]) for box in boxes),
        min(float(box[1]) for box in boxes),
        max(float(box[2]) for box in boxes),
        max(float(box[3]) for box in boxes),
    )


def _candidate_angle(payload: dict[str, Any]) -> float:
    """Estimate the text-row angle in image coordinates, clamped for safety."""

    candidate = _ten_character_candidate(payload)
    if candidate is None:
        return 0.0
    boxes = [box for box in candidate.get("boxes") or [] if len(box) == 4]
    boxes.sort(key=lambda box: (float(box[0]) + float(box[2])) / 2)
    if len(boxes) >= 2:
        first = boxes[0]
        last = boxes[-1]
        first_center = (
            (float(first[0]) + float(first[2])) / 2,
            (float(first[1]) + float(first[3])) / 2,
        )
        last_center = (
            (float(last[0]) + float(last[2])) / 2,
            (float(last[1]) + float(last[3])) / 2,
        )
        delta_x = last_center[0] - first_center[0]
        if abs(delta_x) > 1.0:
            angle = math.degrees(
                math.atan2(last_center[1] - first_center[1], delta_x)
            )
            return max(-12.0, min(12.0, angle))

    source_indices = set(candidate.get("source_indices") or [])
    for item in payload.get("ocr_items") or []:
        if item.get("source_index") not in source_indices:
            continue
        polygon = item.get("polygon") or []
        if len(polygon) < 2:
            continue
        delta_x = float(polygon[1][0]) - float(polygon[0][0])
        if abs(delta_x) <= 1.0:
            continue
        angle = math.degrees(
            math.atan2(
                float(polygon[1][1]) - float(polygon[0][1]),
                delta_x,
            )
        )
        return max(-12.0, min(12.0, angle))
    return 0.0


def _corner_background(image: Image.Image) -> tuple[int, int, int]:
    corners = (
        image.getpixel((0, 0)),
        image.getpixel((max(0, image.width - 1), 0)),
        image.getpixel((0, max(0, image.height - 1))),
        image.getpixel((max(0, image.width - 1), max(0, image.height - 1))),
    )
    return tuple(sum(int(pixel[channel]) for pixel in corners) // 4 for channel in range(3))


def create_container_variant(
    source: Path,
    original_payload: dict[str, Any],
    variant: str,
    destination: Path,
) -> Path | None:
    if variant not in VARIANTS or variant == "original":
        raise ValueError(f"unsupported crop variant: {variant}")
    region = _candidate_region(original_payload)
    if region is None:
        return None

    with Image.open(source) as opened:
        image = ImageOps.exif_transpose(opened).convert("RGB")
        image_width, image_height = image.size
        left, top, right, bottom = region
        text_height = max(1.0, bottom - top)
        text_width = max(1.0, right - left)

        if variant == "check_digit_strip_4x":
            vertical_padding = text_height * 0.25
            crop_box = (
                max(0, int(right)),
                max(0, int(top - vertical_padding)),
                min(image_width, int(right + text_width * 0.30)),
                min(image_height, int(bottom + vertical_padding)),
            )
        elif variant.startswith("right_30_vpad15_"):
            vertical_padding = text_height * 0.15
            crop_box = (
                max(0, int(left)),
                max(0, int(top - vertical_padding)),
                min(image_width, int(right + text_width * 0.30)),
                min(image_height, int(bottom + vertical_padding)),
            )
        elif variant.startswith("right_30_"):
            # Field photos often put the isolated ISO check digit on the right
            # door frame, noticeably farther away from the first ten
            # characters. Preserve the observed number's vertical band so the
            # container type below it (for example 45G1) is not introduced,
            # and extend only the right edge by 30% of the observed width.
            crop_box = (
                max(0, int(left)),
                max(0, int(top)),
                min(image_width, int(right + text_width * 0.30)),
                min(image_height, int(bottom)),
            )
        else:
            # Keep the first experiment intact as a comparison baseline.
            crop_box = (
                max(0, int(left - text_height * 0.8)),
                max(0, int(top - text_height * 1.0)),
                min(image_width, int(right + text_height * 3.0)),
                min(image_height, int(bottom + text_height * 1.0)),
            )
        if crop_box[2] <= crop_box[0] or crop_box[3] <= crop_box[1]:
            return None
        cropped = image.crop(crop_box)
        background = _corner_background(cropped)
        if variant == "right_30_vpad15_deskew_2x":
            cropped = cropped.rotate(
                _candidate_angle(original_payload),
                resample=Image.Resampling.BICUBIC,
                expand=True,
                fillcolor=background,
            )
        if variant == "check_digit_strip_4x":
            border_x = max(4, int(cropped.width * 0.15))
            border_y = max(4, int(cropped.height * 0.15))
            cropped = ImageOps.expand(
                cropped,
                border=(border_x, border_y, border_x, border_y),
                fill=background,
            )
        scale = (
            2
            if variant
            in {
                "context_2x",
                "right_30_2x",
                "right_30_vpad15_2x",
                "right_30_vpad15_deskew_2x",
            }
            else 4
        )
        resized = cropped.resize(
            (cropped.width * scale, cropped.height * scale),
            Image.Resampling.LANCZOS,
        )
        if variant in {"grayscale_4x", "right_30_grayscale_4x"}:
            resized = ImageOps.autocontrast(ImageOps.grayscale(resized)).convert("RGB")
        resized.save(destination, format="PNG")
    return destination


def _compact_single_digit(value: Any) -> str | None:
    compact = re.sub(r"[^A-Z0-9]", "", str(value or "").upper())
    return compact if len(compact) == 1 and compact.isdigit() else None


def _isolated_digit_item(
    payload: dict[str, Any],
    prefix_candidate: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], str] | None:
    items = []
    for item in payload.get("ocr_items") or []:
        digit = _compact_single_digit(item.get("text"))
        if (
            digit is not None
            and float(item.get("score") or 0.0) >= MIN_ISOLATED_DIGIT_SCORE
        ):
            items.append((item, digit))
    if not items:
        return None
    if prefix_candidate is None:
        return max(items, key=lambda pair: float(pair[0].get("score") or 0.0))

    boxes = [box for box in prefix_candidate.get("boxes") or [] if len(box) == 4]
    if not boxes:
        return None
    used_indices = set(prefix_candidate.get("source_indices") or [])
    left = min(float(box[0]) for box in boxes)
    top = min(float(box[1]) for box in boxes)
    right = max(float(box[2]) for box in boxes)
    bottom = max(float(box[3]) for box in boxes)
    text_width = max(1.0, right - left)
    text_height = max(1.0, bottom - top)
    center_y = (top + bottom) / 2
    related: list[tuple[float, float, dict[str, Any], str]] = []
    for item, digit in items:
        if item.get("source_index") in used_indices:
            continue
        box = item.get("box") or []
        if len(box) != 4:
            continue
        item_center_x = (float(box[0]) + float(box[2])) / 2
        item_center_y = (float(box[1]) + float(box[3])) / 2
        gap = float(box[0]) - right
        if item_center_x <= right:
            continue
        if gap > text_width * 0.50:
            continue
        if abs(item_center_y - center_y) > text_height * 1.25:
            continue
        related.append(
            (
                max(0.0, gap),
                -float(item.get("score") or 0.0),
                item,
                digit,
            )
        )
    if not related:
        return None
    _gap, _negative_score, item, digit = min(related, key=lambda row: (row[0], row[1]))
    return item, digit


def combine_isolated_check_digit(
    prefix_payload: dict[str, Any],
    digit_payload: dict[str, Any],
    *,
    digit_is_separate_crop: bool,
    strategy: str,
) -> dict[str, Any]:
    """Create an observed 11-character result from OCR evidence, never from truth.

    The ISO calculation validates an OCR-observed digit but is not used to pick
    or manufacture that digit.
    """

    existing = _container_result(digit_payload)
    if str(existing.get("verification") or "") == "verified":
        return digit_payload

    prefix_candidate = _ten_character_candidate(prefix_payload)
    if prefix_candidate is None:
        return digit_payload
    prefix = str(prefix_candidate.get("observed_value") or "")
    if len(prefix) != 10:
        return digit_payload

    association_candidate = None if digit_is_separate_crop else _ten_character_candidate(
        digit_payload
    )
    digit_match = _isolated_digit_item(digit_payload, association_candidate)
    if digit_match is None:
        return digit_payload
    digit_item, observed_digit = digit_match
    observed = f"{prefix}{observed_digit}"
    suggested = append_check_digit(prefix)
    verified = validate_container_number(observed)
    verification = "verified" if verified else "mismatch"
    validation = "valid" if verified else "invalid"
    prefix_confidence = float(prefix_candidate.get("ocr_confidence") or 0.0)
    digit_confidence = float(digit_item.get("score") or 0.0)
    confidence = round((prefix_confidence + digit_confidence) / 2, 6)
    extraction_score = round(
        min(1.0, confidence * 0.62 + (1.0 if verified else 0.30) * 0.33 + 0.05),
        6,
    )
    digit_box = digit_item.get("box") or []
    candidate = {
        "value": observed,
        "observed_value": observed,
        "suggested_value": suggested,
        "ocr_confidence": confidence,
        "extraction_score": extraction_score,
        "validation": validation,
        "verification": verification,
        "observed_check_digit": observed_digit,
        "calculated_check_digit": suggested[-1],
        "check_digit_source": "observed",
        "inferred": False,
        "corrections": 0,
        "source_texts": [
            *(prefix_candidate.get("source_texts") or []),
            str(digit_item.get("text") or ""),
        ],
        "source_indices": [
            *(prefix_candidate.get("source_indices") or []),
            digit_item.get("source_index"),
        ],
        "boxes": [digit_box] if len(digit_box) == 4 else [],
        "orientation": "original",
        "supporting_orientations": [],
        "selection_reasons": [strategy],
    }
    combined = copy.deepcopy(digit_payload)
    combined.setdefault("results", {})["container_number"] = {
        "target": "container_number",
        "status": "found",
        **candidate,
        "candidates": [candidate],
    }
    combined["postprocessing"] = {
        "strategy": strategy,
        "prefix_observed": prefix,
        "isolated_check_digit": observed_digit,
        "digit_ocr_confidence": digit_confidence,
        "digit_is_separate_crop": digit_is_separate_crop,
    }
    return combined


def _selection_rank(payload: dict[str, Any]) -> tuple[int, float, float]:
    result = _container_result(payload)
    verification = str(result.get("verification") or "")
    rank = {"verified": 3, "unverified": 2, "mismatch": 1}.get(verification, 0)
    return (
        rank,
        float(result.get("extraction_score") or 0.0),
        float(result.get("ocr_confidence") or 0.0),
    )


def select_without_truth(
    payloads: dict[str, dict[str, Any]],
) -> tuple[str, dict[str, Any]]:
    """Select the operational result without looking at the expected answer."""

    return max(payloads.items(), key=lambda item: _selection_rank(item[1]))


def summarize(
    records: list[dict[str, Any]],
    variants: tuple[str, ...] = LEGACY_VARIANTS,
) -> dict[str, Any]:
    eligible = [row for row in records if row.get("api_success") and row.get("expected")]
    baseline_verified = sum(
        bool((row.get("variants") or {}).get("original", {}).get("verified_exact"))
        for row in eligible
    )
    selected_verified = sum(bool(row.get("selected_verified_exact")) for row in eligible)
    by_variant: dict[str, Any] = {}
    for variant in variants:
        rows = [
            (record.get("variants") or {}).get(variant)
            for record in eligible
            if (record.get("variants") or {}).get(variant)
        ]
        by_variant[variant] = {
            "tested": len(rows),
            "verified_exact": sum(bool(row.get("verified_exact")) for row in rows),
            "false_verified": sum(
                row.get("verification") == "verified" and not row.get("observed_exact")
                for row in rows
            ),
            "observed_exact": sum(bool(row.get("observed_exact")) for row in rows),
            "isolated_check_digit_detected": sum(
                bool(row.get("isolated_check_digit")) for row in rows
            ),
            "outcomes": dict(
                sorted(Counter(str(row.get("outcome") or "unknown") for row in rows).items())
            ),
        }
    return {
        "evaluated": len(eligible),
        "baseline_verified_exact": baseline_verified,
        "selected_verified_exact": selected_verified,
        "rescued_verified_exact": selected_verified - baseline_verified,
        "selected_verified_accuracy": (
            round(selected_verified / len(eligible), 4) if eligible else 0.0
        ),
        "selected_false_verified": sum(
            row.get("selected_verification") == "verified"
            and not row.get("selected_verified_exact")
            for row in eligible
        ),
        "selection_counts": dict(
            sorted(Counter(str(row.get("selected_variant")) for row in eligible).items())
        ),
        "by_variant": by_variant,
        "still_unverified_or_wrong": [
            row.get("file_name")
            for row in eligible
            if not row.get("selected_verified_exact")
        ],
    }


def load_retry_file_names(summary_path: Path) -> set[str]:
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    names = ((payload.get("metrics") or {}).get("still_unverified_or_wrong") or [])
    return {str(name) for name in names if str(name).strip()}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Test local crop/upscale strategies for reading the observed ISO 6346 "
            "container check digit through the existing DAS Photo AI API"
        )
    )
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--container-root", type=Path, required=True)
    parser.add_argument("--api-base", default="http://127.0.0.1:8800/api/v1")
    parser.add_argument("--engine", default="paddle_gpu")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--profile",
        choices=tuple(PROFILE_VARIANTS),
        default="legacy",
        help="legacy runs the previous comparison; v032 runs the targeted deskew/digit test",
    )
    parser.add_argument(
        "--retry-summary",
        type=Path,
        help="only test files listed in metrics.still_unverified_or_wrong",
    )
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--limit", type=int, default=0, help="0 runs all container images")
    parser.add_argument("--no-raw", action="store_true")
    parser.add_argument(
        "--save-crops",
        action="store_true",
        help="save generated crop images under output-dir/crops for visual inspection",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if not args.report.is_file():
        raise SystemExit(f"report not found: {args.report}")
    if not args.container_root.is_dir():
        raise SystemExit(f"container image directory not found: {args.container_root}")
    if args.retry_summary and not args.retry_summary.is_file():
        raise SystemExit(f"retry summary not found: {args.retry_summary}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    details_path = args.output_dir / "details.ndjson"
    if details_path.exists():
        raise SystemExit(f"output already contains details.ndjson: {args.output_dir}")

    rows = [
        row for row in load_report_rows(args.report) if row["target"] == "container_number"
    ]
    if args.retry_summary:
        retry_names = load_retry_file_names(args.retry_summary)
        if not retry_names:
            raise SystemExit("retry summary contains no still_unverified_or_wrong files")
        rows = [row for row in rows if row["file_name"] in retry_names]
        matched_names = {row["file_name"] for row in rows}
        missing_names = sorted(retry_names - matched_names)
        if missing_names:
            raise SystemExit(
                "retry summary files are missing from report: " + ", ".join(missing_names)
            )
    if args.limit > 0:
        rows = rows[: args.limit]
    if not rows:
        raise SystemExit("the report contains no container_number rows")

    try:
        health = get_json(f"{args.api_base.rstrip('/')}/health", args.timeout)
        models = get_json(f"{args.api_base.rstrip('/')}/models", args.timeout)
    except Exception as exc:
        raise SystemExit(f"AI service preflight failed: {exc}") from exc
    engines = {item.get("name"): item for item in models.get("engines", [])}
    selected_engine = engines.get(args.engine)
    if not selected_engine or not selected_engine.get("available"):
        raise SystemExit(f"engine is not available: {args.engine}")

    records: list[dict[str, Any]] = []
    selected_variants = PROFILE_VARIANTS[args.profile]
    started = time.perf_counter()
    raw_root = args.output_dir / "raw"
    if not args.no_raw:
        raw_root.mkdir(parents=True, exist_ok=True)

    with details_path.open("a", encoding="utf-8") as detail_file:
        for index, row in enumerate(rows, start=1):
            image_path = args.container_root / row["file_name"]
            record: dict[str, Any] = {
                **row,
                "api_success": False,
                "tested_at": datetime.now(timezone.utc).isoformat(),
                "variants": {},
            }
            if not image_path.is_file():
                record["error"] = f"image not found: {image_path}"
            else:
                payloads: dict[str, dict[str, Any]] = {}
                raw_payloads: dict[str, dict[str, Any]] = {}
                try:
                    original = post_image(
                        args.api_base,
                        image_path,
                        "container_number",
                        args.engine,
                        args.timeout,
                        f"container-check-original-{row['file_stem']}",
                    )
                    payloads["original"] = original
                    raw_payloads["original"] = original
                    with tempfile.TemporaryDirectory(prefix="das-ai-container-crop-") as name:
                        temporary_root = Path(name)
                        for variant in selected_variants[1:]:
                            variant_path = create_container_variant(
                                image_path,
                                original,
                                variant,
                                temporary_root / f"{row['file_stem']}__{variant}.png",
                            )
                            if variant_path is None:
                                continue
                            if args.save_crops:
                                crop_dir = args.output_dir / "crops" / variant
                                crop_dir.mkdir(parents=True, exist_ok=True)
                                shutil.copy2(
                                    variant_path,
                                    crop_dir / f"{row['file_stem']}__{variant}.png",
                                )
                            raw_variant_payload = post_image(
                                args.api_base,
                                variant_path,
                                "container_number",
                                args.engine,
                                args.timeout,
                                f"container-check-{variant}-{row['file_stem']}",
                            )
                            raw_payloads[variant] = raw_variant_payload
                            variant_payload = raw_variant_payload
                            if variant == "check_digit_strip_4x":
                                variant_payload = combine_isolated_check_digit(
                                    original,
                                    variant_payload,
                                    digit_is_separate_crop=True,
                                    strategy="separate_check_digit_crop",
                                )
                            elif variant.startswith("right_30_vpad15_"):
                                variant_payload = combine_isolated_check_digit(
                                    variant_payload,
                                    variant_payload,
                                    digit_is_separate_crop=False,
                                    strategy="right_side_isolated_digit_association",
                                )
                            payloads[variant] = variant_payload

                    for variant, payload in payloads.items():
                        evaluated = evaluate_payload(
                            payload,
                            "container_number",
                            row["expected"],
                        )
                        evaluated["verified_exact"] = bool(
                            evaluated["observed_exact"]
                            and evaluated["verification"] == "verified"
                        )
                        evaluated["isolated_check_digit"] = (
                            payload.get("postprocessing") or {}
                        ).get("isolated_check_digit")
                        evaluated["postprocessing_strategy"] = (
                            payload.get("postprocessing") or {}
                        ).get("strategy")
                        record["variants"][variant] = evaluated
                        if not args.no_raw:
                            raw_dir = raw_root / variant
                            raw_dir.mkdir(parents=True, exist_ok=True)
                            (raw_dir / f"{row['file_stem']}.json").write_text(
                                json.dumps(
                                    raw_payloads.get(variant, payload),
                                    ensure_ascii=False,
                                    indent=2,
                                ),
                                encoding="utf-8",
                            )
                            if payload.get("postprocessing"):
                                processed_dir = args.output_dir / "postprocessed" / variant
                                processed_dir.mkdir(parents=True, exist_ok=True)
                                (processed_dir / f"{row['file_stem']}.json").write_text(
                                    json.dumps(payload, ensure_ascii=False, indent=2),
                                    encoding="utf-8",
                                )

                    selected_variant, selected_payload = select_without_truth(payloads)
                    selected = evaluate_payload(
                        selected_payload,
                        "container_number",
                        row["expected"],
                    )
                    record.update(
                        {
                            "api_success": True,
                            "selected_variant": selected_variant,
                            "selected_observed": selected["observed"],
                            "selected_verification": selected["verification"],
                            "selected_outcome": selected["outcome"],
                            "selected_verified_exact": bool(
                                selected["observed_exact"]
                                and selected["verification"] == "verified"
                            ),
                        }
                    )
                except urllib.error.HTTPError as exc:
                    body = exc.read().decode("utf-8", errors="replace")
                    record["error"] = f"HTTP {exc.code}: {body[:1000]}"
                except Exception as exc:
                    record["error"] = f"{type(exc).__name__}: {exc}"

            records.append(record)
            detail_file.write(json.dumps(record, ensure_ascii=False) + "\n")
            detail_file.flush()
            if record.get("api_success"):
                print(
                    f"[{index}/{len(rows)}] {row['file_name']}: "
                    f"{record['selected_variant']} {record['selected_verification']} "
                    f"{record['selected_observed'] or '-'}",
                    flush=True,
                )
            else:
                print(
                    f"[{index}/{len(rows)}] {row['file_name']}: failed {record.get('error')}",
                    flush=True,
                )

    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "wall_elapsed_seconds": round(time.perf_counter() - started, 3),
        "api_base": args.api_base,
        "requested_engine": args.engine,
        "health": health,
        "selected_engine": selected_engine,
        "container_root": str(args.container_root),
        "profile": args.profile,
        "retry_summary": str(args.retry_summary) if args.retry_summary else None,
        "variants": list(selected_variants),
        "metrics": summarize(records, selected_variants),
    }
    (args.output_dir / "details.json").write_text(
        json.dumps(records, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"output={args.output_dir.resolve()}")
    return 1 if any(not row.get("api_success") for row in records) else 0


if __name__ == "__main__":
    sys.exit(main())
