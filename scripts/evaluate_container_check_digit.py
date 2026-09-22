from __future__ import annotations

import argparse
import json
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

from scripts.evaluate_image_api import (
    evaluate_payload,
    get_json,
    load_report_rows,
    post_image,
)


VARIANTS = (
    "original",
    "context_2x",
    "context_4x",
    "grayscale_4x",
    "right_30_2x",
    "right_30_4x",
    "right_30_grayscale_4x",
)


def _container_result(payload: dict[str, Any]) -> dict[str, Any]:
    return (payload.get("results") or {}).get("container_number") or {}


def _candidate_region(payload: dict[str, Any]) -> tuple[float, float, float, float] | None:
    """Return the box of the ten observed ISO characters, not the suggestion.

    The calculated check digit is intentionally ignored. The crop only uses the
    location of OCR evidence already present in the original image.
    """

    result = _container_result(payload)
    candidates = result.get("candidates") or []
    candidate = next(
        (
            item
            for item in candidates
            if len(str(item.get("observed_value") or "")) == 10
            and item.get("boxes")
        ),
        None,
    )
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

        if variant.startswith("right_30_"):
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
        scale = 2 if variant in {"context_2x", "right_30_2x"} else 4
        resized = cropped.resize(
            (cropped.width * scale, cropped.height * scale),
            Image.Resampling.LANCZOS,
        )
        if variant in {"grayscale_4x", "right_30_grayscale_4x"}:
            resized = ImageOps.autocontrast(ImageOps.grayscale(resized)).convert("RGB")
        resized.save(destination, format="PNG")
    return destination


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


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    eligible = [row for row in records if row.get("api_success") and row.get("expected")]
    baseline_verified = sum(
        bool((row.get("variants") or {}).get("original", {}).get("verified_exact"))
        for row in eligible
    )
    selected_verified = sum(bool(row.get("selected_verified_exact")) for row in eligible)
    by_variant: dict[str, Any] = {}
    for variant in VARIANTS:
        rows = [
            (record.get("variants") or {}).get(variant)
            for record in eligible
            if (record.get("variants") or {}).get(variant)
        ]
        by_variant[variant] = {
            "tested": len(rows),
            "verified_exact": sum(bool(row.get("verified_exact")) for row in rows),
            "observed_exact": sum(bool(row.get("observed_exact")) for row in rows),
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
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--limit", type=int, default=0, help="0 runs all container images")
    parser.add_argument("--no-raw", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if not args.report.is_file():
        raise SystemExit(f"report not found: {args.report}")
    if not args.container_root.is_dir():
        raise SystemExit(f"container image directory not found: {args.container_root}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    details_path = args.output_dir / "details.ndjson"
    if details_path.exists():
        raise SystemExit(f"output already contains details.ndjson: {args.output_dir}")

    rows = [
        row for row in load_report_rows(args.report) if row["target"] == "container_number"
    ]
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
                    with tempfile.TemporaryDirectory(prefix="das-ai-container-crop-") as name:
                        temporary_root = Path(name)
                        for variant in VARIANTS[1:]:
                            variant_path = create_container_variant(
                                image_path,
                                original,
                                variant,
                                temporary_root / f"{row['file_stem']}__{variant}.png",
                            )
                            if variant_path is None:
                                continue
                            payloads[variant] = post_image(
                                args.api_base,
                                variant_path,
                                "container_number",
                                args.engine,
                                args.timeout,
                                f"container-check-{variant}-{row['file_stem']}",
                            )

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
                        record["variants"][variant] = evaluated
                        if not args.no_raw:
                            raw_dir = raw_root / variant
                            raw_dir.mkdir(parents=True, exist_ok=True)
                            (raw_dir / f"{row['file_stem']}.json").write_text(
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
        "variants": list(VARIANTS),
        "metrics": summarize(records),
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
