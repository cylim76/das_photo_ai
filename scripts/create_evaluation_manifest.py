from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from scripts.evaluate_image_api import load_report_rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create a compact evaluation manifest without local database paths"
    )
    parser.add_argument("source_report", type=Path)
    parser.add_argument("output", type=Path)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    rows = [
        {
            "target": row["target"],
            "file_name": row["file_name"],
            "file_stem": row["file_stem"],
            "expected": row["expected"],
        }
        for row in load_report_rows(args.source_report)
    ]
    payload = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"rows={len(rows)}")
    print(f"output={args.output.resolve()}")


if __name__ == "__main__":
    main()
