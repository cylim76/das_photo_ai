from __future__ import annotations

import argparse
import json
import mimetypes
import re
import sqlite3
import sys
import time
import urllib.error
import urllib.request
import uuid
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
TARGET_FOLDERS = {
    "container_number": "container_test",
    "seal_number": "sealno_test",
}


def normalize(value: object) -> str:
    return re.sub(r"[^A-Z0-9]", "", str(value or "").upper())


def _image_files(root: Path) -> list[Path]:
    return sorted(
        path
        for path in root.iterdir()
        if path.is_file()
        and path.suffix.lower() in IMAGE_SUFFIXES
        and "_ocr_" not in path.stem.lower()
    )


def _row_target(row: dict[str, Any]) -> str | None:
    explicit = str(row.get("target") or "").strip()
    if explicit in TARGET_FOLDERS:
        return explicit
    test_type = str(row.get("test_type") or "")
    if "集装箱" in test_type or test_type.lower() == "container":
        return "container_number"
    if "铅封" in test_type or test_type.lower() in {"seal", "sealno"}:
        return "seal_number"
    return None


def load_report_rows(report_path: Path) -> list[dict[str, Any]]:
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    source_rows = payload if isinstance(payload, list) else payload.get("rows", [])
    rows: list[dict[str, Any]] = []
    for source in source_rows:
        if not isinstance(source, dict):
            continue
        target = _row_target(source)
        if target is None:
            continue
        file_name = str(source.get("file_name") or "").strip()
        if not file_name:
            continue
        rows.append(
            {
                "target": target,
                "file_name": file_name,
                "file_stem": str(source.get("file_stem") or Path(file_name).stem),
                "expected": normalize(source.get("expected")),
                "truth_status": str(source.get("source_status") or "report"),
                "cpm_id": source.get("cpm_id"),
                "dataset_path": source.get("dataset_path"),
            }
        )
    return rows


def _add_truth(index: dict[str, list[dict[str, Any]]], key: object, row: dict[str, Any]) -> None:
    normalized = normalize(key)
    if normalized:
        index[normalized].append(row)


def load_database_rows(
    database_path: Path,
    container_root: Path,
    seal_root: Path,
) -> list[dict[str, Any]]:
    index: dict[str, list[dict[str, Any]]] = defaultdict(list)
    uri = database_path.resolve().as_uri() + "?mode=ro"
    with sqlite3.connect(uri, uri=True) as connection:
        connection.row_factory = sqlite3.Row
        source_rows = connection.execute(
            """
            SELECT p.id AS photo_id, p.cpm_id, p.photo_label,
                   p.local_path, p.relative_path,
                   c.container_no, c.seal_no
            FROM photos p
            LEFT JOIN containers c ON c.cpm_id = p.cpm_id
            """
        ).fetchall()

    for source in source_rows:
        row = dict(source)
        _add_truth(index, row.get("photo_label"), row)
        for field in ("local_path", "relative_path"):
            value = row.get(field)
            if value:
                _add_truth(index, Path(str(value)).stem, row)

    rows: list[dict[str, Any]] = []
    for target, root in (
        ("container_number", container_root),
        ("seal_number", seal_root),
    ):
        truth_field = "container_no" if target == "container_number" else "seal_no"
        for image_path in _image_files(root):
            matches = index.get(normalize(image_path.stem), [])
            values = sorted(
                {
                    normalize(match.get(truth_field))
                    for match in matches
                    if normalize(match.get(truth_field))
                }
            )
            cpm_ids = sorted(
                {match.get("cpm_id") for match in matches if match.get("cpm_id") is not None}
            )
            if len(values) == 1:
                expected = values[0]
                truth_status = "matched"
            elif not values:
                expected = ""
                truth_status = "missing"
            else:
                expected = ""
                truth_status = "ambiguous:" + "|".join(values)
            rows.append(
                {
                    "target": target,
                    "file_name": image_path.name,
                    "file_stem": image_path.stem,
                    "expected": expected,
                    "truth_status": truth_status,
                    "cpm_id": cpm_ids[0] if len(cpm_ids) == 1 else None,
                    "dataset_path": None,
                }
            )
    return rows


def _multipart_body(
    fields: dict[str, str],
    image_path: Path,
) -> tuple[bytes, str]:
    boundary = f"----das-photo-ai-{uuid.uuid4().hex}"
    chunks: list[bytes] = []
    for name, value in fields.items():
        chunks.extend(
            [
                f"--{boundary}\r\n".encode(),
                f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode(),
                value.encode("utf-8"),
                b"\r\n",
            ]
        )
    content_type = mimetypes.guess_type(image_path.name)[0] or "application/octet-stream"
    chunks.extend(
        [
            f"--{boundary}\r\n".encode(),
            (
                f'Content-Disposition: form-data; name="image"; '
                f'filename="{image_path.name}"\r\n'
            ).encode("utf-8"),
            f"Content-Type: {content_type}\r\n\r\n".encode(),
            image_path.read_bytes(),
            b"\r\n",
            f"--{boundary}--\r\n".encode(),
        ]
    )
    return b"".join(chunks), f"multipart/form-data; boundary={boundary}"


def get_json(url: str, timeout: float) -> dict[str, Any]:
    request = urllib.request.Request(url, method="GET", headers={"Accept": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def post_image(
    api_base: str,
    image_path: Path,
    target: str,
    engine: str,
    timeout: float,
    request_id: str,
) -> dict[str, Any]:
    body, content_type = _multipart_body(
        {"engine": engine, "targets": target, "request_id": request_id},
        image_path,
    )
    request = urllib.request.Request(
        f"{api_base.rstrip('/')}/recognize/image",
        data=body,
        method="POST",
        headers={
            "Accept": "application/json",
            "Content-Type": content_type,
            "Content-Length": str(len(body)),
        },
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def evaluate_payload(
    payload: dict[str, Any],
    target: str,
    expected: str,
) -> dict[str, Any]:
    result = (payload.get("results") or {}).get(target) or {}
    candidates = result.get("candidates") or []
    observed = normalize(result.get("observed_value") or result.get("value"))
    suggested = normalize(result.get("suggested_value") or observed)
    candidate_values: list[str] = []
    for candidate in candidates:
        for value in (
            candidate.get("observed_value"),
            candidate.get("suggested_value"),
            candidate.get("value"),
        ):
            normalized = normalize(value)
            if normalized and normalized not in candidate_values:
                candidate_values.append(normalized)

    observed_exact = bool(expected and observed == expected)
    suggested_exact = bool(expected and suggested == expected)
    candidate_contains = bool(expected and expected in candidate_values)
    verification = str(result.get("verification") or "not_applicable")
    postprocessing = result.get("postprocessing") or {}

    if not expected:
        outcome = "missing_truth"
    elif target == "container_number" and observed_exact and verification == "verified":
        outcome = "verified_exact"
    elif target == "container_number" and suggested_exact:
        outcome = "inferred_exact"
    elif target == "seal_number" and observed_exact:
        outcome = "exact"
    elif candidate_contains:
        outcome = "candidate_hit"
    elif result.get("status") == "not_found" or not observed:
        outcome = "not_found"
    else:
        outcome = "wrong"

    return {
        "outcome": outcome,
        "status": result.get("status"),
        "observed": observed,
        "suggested": suggested,
        "observed_exact": observed_exact,
        "suggested_exact": suggested_exact,
        "candidate_contains": candidate_contains,
        "verification": verification,
        "validation": result.get("validation"),
        "ocr_confidence": result.get("ocr_confidence"),
        "extraction_score": result.get("extraction_score"),
        "observed_check_digit": result.get("observed_check_digit"),
        "calculated_check_digit": result.get("calculated_check_digit"),
        "candidate_values": candidate_values,
        "ocr_text_count": len(payload.get("ocr_items") or []),
        "service_elapsed_ms": payload.get("elapsed_ms"),
        "service_version": payload.get("service_version"),
        "extractor_version": payload.get("extractor_version"),
        "engine": payload.get("engine"),
        "ocr_source": payload.get("ocr_source"),
        "request_id": payload.get("request_id"),
        "postprocessing_status": postprocessing.get("status"),
        "postprocessing_strategy": postprocessing.get("strategy"),
        "postprocessing_model": postprocessing.get("model_name"),
    }


def _percent(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 4) if denominator else 0.0


def _percentile(values: Iterable[float], percentile: float) -> float | None:
    ordered = sorted(values)
    if not ordered:
        return None
    index = max(0, min(len(ordered) - 1, int((len(ordered) * percentile) + 0.999999) - 1))
    return round(ordered[index], 3)


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for target in TARGET_FOLDERS:
        rows = [record for record in records if record.get("target") == target]
        counts = Counter(str(row.get("outcome") or "unknown") for row in rows)
        success = [row for row in rows if row.get("api_success")]
        eligible = [row for row in success if row.get("expected")]
        elapsed = [
            float(row["service_elapsed_ms"])
            for row in success
            if isinstance(row.get("service_elapsed_ms"), (int, float))
        ]
        verification = Counter(str(row.get("verification") or "unknown") for row in success)
        postprocessing = Counter(
            str(row.get("postprocessing_status") or "not_reported") for row in success
        )
        observed_exact = sum(bool(row.get("observed_exact")) for row in eligible)
        verified_exact = sum(
            bool(row.get("observed_exact")) and row.get("verification") == "verified"
            for row in eligible
        )
        false_verified = sum(
            not row.get("observed_exact") and row.get("verification") == "verified"
            for row in eligible
        )
        suggested_exact = sum(bool(row.get("suggested_exact")) for row in eligible)
        candidate_contains = sum(bool(row.get("candidate_contains")) for row in eligible)
        output[target] = {
            "total": len(rows),
            "api_success": len(success),
            "api_errors": sum(not row.get("api_success") and not row.get("missing_image") for row in rows),
            "missing_images": sum(bool(row.get("missing_image")) for row in rows),
            "missing_truth": sum(not row.get("expected") for row in rows),
            "evaluated": len(eligible),
            "outcomes": dict(sorted(counts.items())),
            "verification": dict(sorted(verification.items())),
            "postprocessing": dict(sorted(postprocessing.items())),
            "observed_top1_exact": observed_exact,
            "observed_top1_accuracy": _percent(observed_exact, len(eligible)),
            "verified_exact": verified_exact,
            "verified_exact_accuracy": _percent(verified_exact, len(eligible)),
            "false_verified": false_verified,
            "suggested_top1_exact": suggested_exact,
            "suggested_top1_accuracy": _percent(suggested_exact, len(eligible)),
            "candidate_contains": candidate_contains,
            "candidate_recall": _percent(candidate_contains, len(eligible)),
            "service_elapsed_ms": {
                "count": len(elapsed),
                "average": round(sum(elapsed) / len(elapsed), 3) if elapsed else None,
                "median": _percentile(elapsed, 0.5),
                "p95": _percentile(elapsed, 0.95),
                "max": round(max(elapsed), 3) if elapsed else None,
            },
        }
    return output


def _record_key(record: dict[str, Any]) -> str:
    return f"{record.get('target')}:{record.get('file_name')}"


def _load_previous_records(path: Path) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return records
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        records[_record_key(record)] = record
    return records


def _print_progress(index: int, total: int, record: dict[str, Any]) -> None:
    target = "箱号" if record["target"] == "container_number" else "铅封号"
    if record.get("api_success"):
        detail = f"{record.get('outcome')} 结果={record.get('suggested') or record.get('observed') or '-'}"
        elapsed = record.get("service_elapsed_ms")
        if isinstance(elapsed, (int, float)):
            detail += f" {elapsed:.0f}ms"
    else:
        detail = f"失败: {record.get('error') or 'unknown error'}"
    print(f"[{index}/{total}] {target} {record['file_name']}: {detail}", flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run an end-to-end image baseline against the DAS Photo AI HTTP API"
    )
    truth = parser.add_mutually_exclusive_group(required=True)
    truth.add_argument("--report", type=Path, help="Existing ocr_eval_data.json")
    truth.add_argument("--database", type=Path, help="Read-only DAS Photo SQLite database")
    parser.add_argument("--container-root", type=Path, required=True)
    parser.add_argument("--seal-root", type=Path, required=True)
    parser.add_argument("--api-base", default="http://127.0.0.1:8800/api/v1")
    parser.add_argument("--engine", default="paddle_gpu")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--limit", type=int, default=0, help="0 runs all images")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--retry-failures", action="store_true")
    parser.add_argument("--no-raw", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    for root in (args.container_root, args.seal_root):
        if not root.is_dir():
            raise SystemExit(f"image directory not found: {root}")
    if args.report and not args.report.is_file():
        raise SystemExit(f"report not found: {args.report}")
    if args.database and not args.database.is_file():
        raise SystemExit(f"database not found: {args.database}")

    timestamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir or Path("evaluation_runs") / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    details_path = output_dir / "details.ndjson"
    if details_path.exists() and not args.resume:
        raise SystemExit(f"output already contains details.ndjson; use --resume: {output_dir}")
    raw_root = output_dir / "raw"
    if not args.no_raw:
        raw_root.mkdir(parents=True, exist_ok=True)

    rows = (
        load_report_rows(args.report)
        if args.report
        else load_database_rows(args.database, args.container_root, args.seal_root)
    )
    if args.limit > 0:
        rows = rows[: args.limit]
    previous = _load_previous_records(details_path) if args.resume else {}

    try:
        health = get_json(f"{args.api_base.rstrip('/')}/health", args.timeout)
        models = get_json(f"{args.api_base.rstrip('/')}/models", args.timeout)
    except Exception as exc:
        raise SystemExit(f"AI service preflight failed: {exc}") from exc
    engines = {item.get("name"): item for item in models.get("engines", [])}
    selected_engine = engines.get(args.engine)
    if not selected_engine or not selected_engine.get("available"):
        raise SystemExit(f"engine is not available: {args.engine}")

    latest = dict(previous)
    started = time.perf_counter()
    with details_path.open("a", encoding="utf-8") as detail_file:
        for index, row in enumerate(rows, start=1):
            key = _record_key(row)
            old = previous.get(key)
            if old and (old.get("api_success") or not args.retry_failures):
                _print_progress(index, len(rows), old)
                continue

            target = row["target"]
            root = args.container_root if target == "container_number" else args.seal_root
            image_path = root / row["file_name"]
            record = {
                **row,
                "api_success": False,
                "missing_image": False,
                "tested_at": datetime.now(timezone.utc).isoformat(),
            }
            if not image_path.is_file():
                record["missing_image"] = True
                record["error"] = f"image not found: {image_path}"
            else:
                request_id = f"eval-{target}-{row['file_stem']}"
                wall_started = time.perf_counter()
                try:
                    payload = post_image(
                        args.api_base,
                        image_path,
                        target,
                        args.engine,
                        args.timeout,
                        request_id,
                    )
                    record.update(evaluate_payload(payload, target, row["expected"]))
                    record["api_success"] = True
                    record["wall_elapsed_ms"] = round(
                        (time.perf_counter() - wall_started) * 1000,
                        3,
                    )
                    if not args.no_raw:
                        raw_dir = raw_root / target
                        raw_dir.mkdir(parents=True, exist_ok=True)
                        raw_path = raw_dir / f"{row['file_stem']}.json"
                        raw_path.write_text(
                            json.dumps(payload, ensure_ascii=False, indent=2),
                            encoding="utf-8",
                        )
                        record["raw_response"] = str(raw_path.relative_to(output_dir))
                except urllib.error.HTTPError as exc:
                    body = exc.read().decode("utf-8", errors="replace")
                    record["error"] = f"HTTP {exc.code}: {body[:1000]}"
                except Exception as exc:
                    record["error"] = f"{type(exc).__name__}: {exc}"

            detail_file.write(json.dumps(record, ensure_ascii=False) + "\n")
            detail_file.flush()
            latest[key] = record
            _print_progress(index, len(rows), record)

    ordered = [latest[_record_key(row)] for row in rows if _record_key(row) in latest]
    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "wall_elapsed_seconds": round(time.perf_counter() - started, 3),
        "api_base": args.api_base,
        "requested_engine": args.engine,
        "health": health,
        "selected_engine": selected_engine,
        "test_roots": {
            "container_number": str(args.container_root),
            "seal_number": str(args.seal_root),
        },
        "metrics": summarize(ordered),
    }
    (output_dir / "details.json").write_text(
        json.dumps(ordered, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"output={output_dir.resolve()}")

    has_failures = any(
        not record.get("api_success") or not record.get("expected") for record in ordered
    )
    return 1 if has_failures else 0


if __name__ == "__main__":
    sys.exit(main())
