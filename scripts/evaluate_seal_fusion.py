from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from app.pipelines.common import ExtractionCandidate
from app.pipelines.seal_fusion import fuse_seal_candidates


ORIENTATIONS = ("original", "cw90", "ccw90")


def _candidate_from_payload(value: dict[str, Any], orientation: str) -> ExtractionCandidate:
    return ExtractionCandidate(
        value=str(value.get("value") or ""),
        observed_value=str(value.get("observed_value") or value.get("value") or ""),
        suggested_value=value.get("suggested_value"),
        ocr_confidence=float(value.get("ocr_confidence") or 0.0),
        extraction_score=float(value.get("extraction_score") or 0.0),
        validation=value.get("validation"),
        verification=str(value.get("verification") or "not_applicable"),
        observed_check_digit=value.get("observed_check_digit"),
        calculated_check_digit=value.get("calculated_check_digit"),
        check_digit_source=value.get("check_digit_source"),
        inferred=bool(value.get("inferred")),
        corrections=int(value.get("corrections") or 0),
        source_texts=tuple(str(item) for item in value.get("source_texts") or []),
        source_indices=tuple(int(item) for item in value.get("source_indices") or []),
        boxes=tuple(tuple(float(part) for part in box) for box in value.get("boxes") or []),
        orientation=orientation,
    )


def evaluate_run(run_dir: Path, max_candidates: int = 10) -> dict[str, Any]:
    details = json.loads((run_dir / "details.json").read_text(encoding="utf-8"))
    expected_by_file: dict[str, str] = {}
    for row in details:
        expected_by_file[str(row["file_name"])] = str(row.get("expected") or "")

    rows: list[dict[str, Any]] = []
    for file_name, expected in sorted(expected_by_file.items()):
        candidates_by_orientation: dict[str, list[ExtractionCandidate]] = {}
        for orientation in ORIENTATIONS:
            raw_path = run_dir / "raw" / orientation / f"{Path(file_name).stem}.json"
            payload = json.loads(raw_path.read_text(encoding="utf-8"))
            raw_candidates = (
                (payload.get("results") or {}).get("seal_number") or {}
            ).get("candidates") or []
            candidates_by_orientation[orientation] = [
                _candidate_from_payload(candidate, orientation)
                for candidate in raw_candidates
            ]
        fused = fuse_seal_candidates(candidates_by_orientation, max_candidates)
        best = fused[0] if fused else None
        rows.append(
            {
                "file_name": file_name,
                "expected": expected,
                "observed": best.observed_value if best else "",
                "exact": bool(best and best.observed_value == expected),
                "orientation": best.orientation if best else None,
                "score": best.extraction_score if best else None,
                "supporting_orientations": list(best.supporting_orientations) if best else [],
                "selection_reasons": list(best.selection_reasons) if best else [],
                "candidate_values": [candidate.value for candidate in fused],
                "candidate_contains": any(candidate.value == expected for candidate in fused),
            }
        )

    exact = sum(row["exact"] for row in rows)
    contains = sum(row["candidate_contains"] for row in rows)
    return {
        "total": len(rows),
        "top1_exact": exact,
        "top1_accuracy": round(exact / len(rows), 4) if rows else 0.0,
        "candidate_contains": contains,
        "candidate_recall": round(contains / len(rows), 4) if rows else 0.0,
        "misses": [row for row in rows if not row["exact"]],
        "rows": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate seal orientation fusion offline")
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--max-candidates", type=int, default=10)
    args = parser.parse_args()

    result = evaluate_run(args.run_dir, args.max_candidates)
    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
