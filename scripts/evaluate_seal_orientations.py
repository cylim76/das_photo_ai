from __future__ import annotations

import argparse
import csv
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
except ImportError as exc:  # pragma: no cover - exercised on machines without Pillow
    raise SystemExit(
        "Pillow is required for the orientation experiment. "
        "Install requirements-dev.txt or run this script in the DAS Photo AI image."
    ) from exc

from scripts.evaluate_image_api import (
    evaluate_payload,
    get_json,
    load_report_rows,
    post_image,
)


ORIENTATIONS = ("original", "cw90", "ccw90")


def _record_key(file_name: str, orientation: str) -> str:
    return f"{file_name}:{orientation}"


def _load_previous_records(path: Path) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return records
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        file_name = str(record.get("file_name") or "")
        orientation = str(record.get("orientation") or "")
        if file_name and orientation in ORIENTATIONS:
            records[_record_key(file_name, orientation)] = record
    return records


def create_oriented_image(
    source: Path,
    orientation: str,
    temporary_root: Path,
) -> Path:
    if orientation == "original":
        return source
    if orientation not in ORIENTATIONS:
        raise ValueError(f"unsupported orientation: {orientation}")

    destination = temporary_root / f"{source.stem}__{orientation}.png"
    transpose = (
        Image.Transpose.ROTATE_270
        if orientation == "cw90"
        else Image.Transpose.ROTATE_90
    )
    with Image.open(source) as opened:
        normalized = ImageOps.exif_transpose(opened)
        rotated = normalized.transpose(transpose)
        rotated.save(destination, format="PNG")
    return destination


def summarize_orientation_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    by_orientation: dict[str, Any] = {}
    indexed: dict[str, dict[str, dict[str, Any]]] = {}

    for orientation in ORIENTATIONS:
        rows = [row for row in records if row.get("orientation") == orientation]
        successful = [row for row in rows if row.get("api_success")]
        eligible = [row for row in successful if row.get("expected")]
        exact = [row for row in eligible if row.get("observed_exact")]
        candidate_hits = [row for row in eligible if row.get("candidate_contains")]
        elapsed = [
            float(row["service_elapsed_ms"])
            for row in successful
            if isinstance(row.get("service_elapsed_ms"), (int, float))
        ]
        by_orientation[orientation] = {
            "total": len(rows),
            "api_success": len(successful),
            "api_errors": len(rows) - len(successful),
            "evaluated": len(eligible),
            "outcomes": dict(
                sorted(Counter(str(row.get("outcome") or "unknown") for row in rows).items())
            ),
            "top1_exact": len(exact),
            "top1_accuracy": round(len(exact) / len(eligible), 4) if eligible else 0.0,
            "candidate_contains": len(candidate_hits),
            "candidate_recall": (
                round(len(candidate_hits) / len(eligible), 4) if eligible else 0.0
            ),
            "average_service_elapsed_ms": (
                round(sum(elapsed) / len(elapsed), 3) if elapsed else None
            ),
        }
        for row in eligible:
            indexed.setdefault(str(row["file_name"]), {})[orientation] = row

    original_exact = {
        file_name
        for file_name, variants in indexed.items()
        if variants.get("original", {}).get("observed_exact")
    }
    comparisons: dict[str, Any] = {}
    for orientation in ("cw90", "ccw90"):
        orientation_exact = {
            file_name
            for file_name, variants in indexed.items()
            if variants.get(orientation, {}).get("observed_exact")
        }
        comparisons[orientation] = {
            "rescued_vs_original": sorted(orientation_exact - original_exact),
            "rescued_count": len(orientation_exact - original_exact),
            "regressed_vs_original": sorted(original_exact - orientation_exact),
            "regressed_count": len(original_exact - orientation_exact),
            "original_or_rotated_exact": len(original_exact | orientation_exact),
        }

    any_orientation_exact = {
        file_name
        for file_name, variants in indexed.items()
        if any(variants.get(name, {}).get("observed_exact") for name in ORIENTATIONS)
    }
    evaluated_files = len(indexed)
    return {
        "by_orientation": by_orientation,
        "comparison": comparisons,
        "combined": {
            "evaluated_files": evaluated_files,
            "any_orientation_exact": len(any_orientation_exact),
            "any_orientation_accuracy": (
                round(len(any_orientation_exact) / evaluated_files, 4)
                if evaluated_files
                else 0.0
            ),
            "still_not_exact": sorted(set(indexed) - any_orientation_exact),
        },
    }


def write_comparison_csv(path: Path, records: list[dict[str, Any]]) -> None:
    indexed: dict[str, dict[str, dict[str, Any]]] = {}
    for record in records:
        indexed.setdefault(str(record["file_name"]), {})[
            str(record["orientation"])
        ] = record

    fields = ["file_name", "expected"]
    for orientation in ORIENTATIONS:
        fields.extend(
            [
                f"{orientation}_observed",
                f"{orientation}_outcome",
                f"{orientation}_exact",
                f"{orientation}_elapsed_ms",
            ]
        )
    fields.append("successful_orientations")

    with path.open("w", encoding="utf-8-sig", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        for file_name in sorted(indexed):
            variants = indexed[file_name]
            first = next(iter(variants.values()))
            row: dict[str, Any] = {
                "file_name": file_name,
                "expected": first.get("expected"),
            }
            successful_orientations: list[str] = []
            for orientation in ORIENTATIONS:
                result = variants.get(orientation, {})
                row[f"{orientation}_observed"] = result.get("observed")
                row[f"{orientation}_outcome"] = result.get("outcome")
                row[f"{orientation}_exact"] = bool(result.get("observed_exact"))
                row[f"{orientation}_elapsed_ms"] = result.get("service_elapsed_ms")
                if result.get("observed_exact"):
                    successful_orientations.append(orientation)
            row["successful_orientations"] = ",".join(successful_orientations)
            writer.writerow(row)


def _print_progress(index: int, total: int, record: dict[str, Any]) -> None:
    if record.get("api_success"):
        result = record.get("observed") or "-"
        detail = f"{record.get('outcome')} result={result}"
    else:
        detail = f"failed: {record.get('error') or 'unknown error'}"
    print(
        f"[{index}/{total}] {record['file_name']} {record['orientation']}: {detail}",
        flush=True,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compare original, clockwise 90 degree and counter-clockwise 90 degree "
            "seal-number OCR through the existing DAS Photo AI API"
        )
    )
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--seal-root", type=Path, required=True)
    parser.add_argument("--api-base", default="http://127.0.0.1:8800/api/v1")
    parser.add_argument("--engine", default="paddle_gpu")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--limit", type=int, default=0, help="0 runs all seal images")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--retry-failures", action="store_true")
    parser.add_argument("--no-raw", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if not args.report.is_file():
        raise SystemExit(f"report not found: {args.report}")
    if not args.seal_root.is_dir():
        raise SystemExit(f"seal image directory not found: {args.seal_root}")

    rows = [
        row for row in load_report_rows(args.report) if row["target"] == "seal_number"
    ]
    if args.limit > 0:
        rows = rows[: args.limit]
    if not rows:
        raise SystemExit("the report contains no seal_number rows")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    details_path = args.output_dir / "details.ndjson"
    if details_path.exists() and not args.resume:
        raise SystemExit(
            f"output already contains details.ndjson; use --resume: {args.output_dir}"
        )
    raw_root = args.output_dir / "raw"
    if not args.no_raw:
        raw_root.mkdir(parents=True, exist_ok=True)

    try:
        health = get_json(f"{args.api_base.rstrip('/')}/health", args.timeout)
        models = get_json(f"{args.api_base.rstrip('/')}/models", args.timeout)
    except Exception as exc:
        raise SystemExit(f"AI service preflight failed: {exc}") from exc
    engines = {item.get("name"): item for item in models.get("engines", [])}
    selected_engine = engines.get(args.engine)
    if not selected_engine or not selected_engine.get("available"):
        raise SystemExit(f"engine is not available: {args.engine}")

    previous = _load_previous_records(details_path) if args.resume else {}
    latest = dict(previous)
    planned = [(row, orientation) for row in rows for orientation in ORIENTATIONS]
    started = time.perf_counter()

    with details_path.open("a", encoding="utf-8") as detail_file:
        for index, (row, orientation) in enumerate(planned, start=1):
            key = _record_key(row["file_name"], orientation)
            old = previous.get(key)
            if old and (old.get("api_success") or not args.retry_failures):
                _print_progress(index, len(planned), old)
                continue

            image_path = args.seal_root / row["file_name"]
            record: dict[str, Any] = {
                **row,
                "orientation": orientation,
                "api_success": False,
                "missing_image": False,
                "tested_at": datetime.now(timezone.utc).isoformat(),
            }
            if not image_path.is_file():
                record["missing_image"] = True
                record["error"] = f"image not found: {image_path}"
            else:
                request_id = f"seal-orientation-{orientation}-{row['file_stem']}"
                wall_started = time.perf_counter()
                try:
                    with tempfile.TemporaryDirectory(prefix="das-ai-orientation-") as name:
                        request_image = create_oriented_image(
                            image_path,
                            orientation,
                            Path(name),
                        )
                        payload = post_image(
                            args.api_base,
                            request_image,
                            "seal_number",
                            args.engine,
                            args.timeout,
                            request_id,
                        )
                    record.update(
                        evaluate_payload(payload, "seal_number", row["expected"])
                    )
                    record["api_success"] = True
                    record["wall_elapsed_ms"] = round(
                        (time.perf_counter() - wall_started) * 1000,
                        3,
                    )
                    if not args.no_raw:
                        raw_dir = raw_root / orientation
                        raw_dir.mkdir(parents=True, exist_ok=True)
                        raw_path = raw_dir / f"{row['file_stem']}.json"
                        raw_path.write_text(
                            json.dumps(payload, ensure_ascii=False, indent=2),
                            encoding="utf-8",
                        )
                        record["raw_response"] = str(
                            raw_path.relative_to(args.output_dir)
                        )
                except urllib.error.HTTPError as exc:
                    body = exc.read().decode("utf-8", errors="replace")
                    record["error"] = f"HTTP {exc.code}: {body[:1000]}"
                except Exception as exc:
                    record["error"] = f"{type(exc).__name__}: {exc}"

            detail_file.write(json.dumps(record, ensure_ascii=False) + "\n")
            detail_file.flush()
            latest[key] = record
            _print_progress(index, len(planned), record)

    ordered = [
        latest[_record_key(row["file_name"], orientation)]
        for row, orientation in planned
        if _record_key(row["file_name"], orientation) in latest
    ]
    metrics = summarize_orientation_records(ordered)
    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "wall_elapsed_seconds": round(time.perf_counter() - started, 3),
        "api_base": args.api_base,
        "requested_engine": args.engine,
        "health": health,
        "selected_engine": selected_engine,
        "seal_root": str(args.seal_root),
        "orientations": list(ORIENTATIONS),
        "metrics": metrics,
    }
    (args.output_dir / "details.json").write_text(
        json.dumps(ordered, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_comparison_csv(args.output_dir / "comparison.csv", ordered)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"output={args.output_dir.resolve()}")

    has_operational_failures = any(
        not record.get("api_success") or not record.get("expected") for record in ordered
    )
    return 1 if has_operational_failures else 0


if __name__ == "__main__":
    sys.exit(main())
