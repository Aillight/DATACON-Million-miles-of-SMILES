from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.aggregation import aggregate_extraction_rows
from backend.nano_aggregation import aggregate_nanozyme_rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate extracted ChemX rows into a clean CSV.")
    parser.add_argument("inputs", nargs="+", type=Path, help="Input CSV, JSON, or JSONL files.")
    parser.add_argument("--output", type=Path, default=Path("outputs/extractions/clean.csv"))
    parser.add_argument("--conflicts", type=Path, default=Path("outputs/extractions/conflicts.csv"))
    parser.add_argument("--kind", choices=("small-molecule", "nanozyme"), default="small-molecule")
    parser.add_argument("--value-decimals", type=int, default=3)
    args = parser.parse_args()

    rows = read_extraction_rows(args.inputs)
    if args.kind == "nanozyme":
        result = aggregate_nanozyme_rows(rows, value_decimals=args.value_decimals)
    else:
        result = aggregate_extraction_rows(rows, value_decimals=args.value_decimals)
    result.write_csv(args.output, args.conflicts)

    print(f"input_rows={len(rows)}")
    print(f"clean_rows={len(result.clean)}")
    print(f"conflicts={len(result.conflicts)}")
    print(f"output={args.output}")
    print(f"conflicts_output={args.conflicts}")


def read_extraction_rows(paths: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        suffix = path.suffix.lower()
        if suffix == ".csv":
            rows.extend(read_csv_rows(path))
        elif suffix == ".jsonl":
            rows.extend(read_jsonl_rows(path))
        elif suffix == ".json":
            rows.extend(read_json_rows(path))
        else:
            raise ValueError(f"Unsupported input file type: {path}")
    return rows


def read_csv_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        return [dict(row) for row in reader]


def read_jsonl_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            rows.extend(normalize_json_payload(json.loads(stripped), source=path))
    return rows


def read_json_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return normalize_json_payload(json.load(handle), source=path)


def normalize_json_payload(payload: Any, source: Path) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [row for row in payload if isinstance(row, dict)]
    if isinstance(payload, dict):
        for key in ("validated_objects", "extracted_objects", "rows", "data"):
            value = payload.get(key)
            if isinstance(value, list):
                return [row for row in value if isinstance(row, dict)]
        return [payload]
    raise ValueError(f"Unsupported JSON payload in {source}")


if __name__ == "__main__":
    main()
