from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

from app.engines.json_adapter import adapt_ocr_json
from app.pipelines.container_number import extract_container_numbers
from app.pipelines.seal_number import extract_seal_numbers


def normalize(value: object) -> str:
    return re.sub(r"[^A-Z0-9]", "", str(value or "").upper())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate DAS Photo AI extractors against an OCR evaluation report"
    )
    parser.add_argument("report", type=Path, help="ocr_eval_data.json containing rows")
    parser.add_argument(
        "--ocr-root",
        type=Path,
        required=True,
        help="Directory containing container_test and sealno_test",
    )
    parser.add_argument("--max-candidates", type=int, default=10)
    parser.add_argument("--show-errors", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    report = json.loads(args.report.read_text(encoding="utf-8"))
    rows = report.get("rows", [])
    summaries: dict[str, Counter[str]] = {
        "container_number": Counter(),
        "seal_number": Counter(),
    }
    errors: list[dict[str, str]] = []

    for row in rows:
        is_container = row.get("test_type") == "集装箱号"
        target = "container_number" if is_container else "seal_number"
        folder = "container_test" if is_container else "sealno_test"
        stem = str(row.get("file_stem") or Path(str(row.get("file_name", ""))).stem)
        json_path = args.ocr_root / folder / f"{stem}_res.json"
        expected = normalize(row.get("expected"))
        summary = summaries[target]
        summary["total"] += 1
        if not expected:
            summary["missing_truth"] += 1
            continue
        if not json_path.exists():
            summary["missing_json"] += 1
            errors.append(
                {"target": target, "file": stem, "expected": expected, "actual": "<missing json>"}
            )
            continue

        payload = json.loads(json_path.read_text(encoding="utf-8"))
        document = adapt_ocr_json(payload)
        candidates = (
            extract_container_numbers(document, args.max_candidates)
            if is_container
            else extract_seal_numbers(document, args.max_candidates)
        )
        observed_values = [candidate.observed_value for candidate in candidates]
        suggested_values = [candidate.suggested_value or candidate.value for candidate in candidates]
        actual = suggested_values[0] if suggested_values else ""
        if candidates:
            summary[f"verification_{candidates[0].verification}"] += 1
            if candidates[0].observed_value == expected:
                summary["observed_top1_exact"] += 1
        if actual == expected:
            summary["suggested_top1_exact"] += 1
        else:
            summary["top1_wrong"] += 1
            errors.append(
                {
                    "target": target,
                    "file": stem,
                    "expected": expected,
                    "actual": actual,
                    "observed_candidates": " | ".join(observed_values),
                    "suggested_candidates": " | ".join(suggested_values),
                }
            )
        if expected in suggested_values:
            summary["suggested_candidates_contains"] += 1
        if suggested_values:
            summary["found_any"] += 1

    output: dict[str, dict[str, int | float]] = {}
    for target, summary in summaries.items():
        evaluated = summary["total"] - summary["missing_truth"] - summary["missing_json"]
        output[target] = dict(summary)
        output[target]["evaluated"] = evaluated
        output[target]["suggested_top1_accuracy"] = (
            round(summary["suggested_top1_exact"] / evaluated, 4) if evaluated else 0.0
        )
        output[target]["observed_top1_accuracy"] = (
            round(summary["observed_top1_exact"] / evaluated, 4) if evaluated else 0.0
        )
        output[target]["suggested_candidate_recall"] = (
            round(summary["suggested_candidates_contains"] / evaluated, 4)
            if evaluated
            else 0.0
        )

    print(json.dumps(output, ensure_ascii=False, indent=2))
    if args.show_errors:
        print(json.dumps(errors, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
