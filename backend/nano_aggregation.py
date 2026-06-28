from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from agent.nano_extraction_graph import validate_material_formula
from backend.quality import add_quality_columns


TABLE_SOURCE_TYPES = {"table", "docling_table", "camelot_table"}
TEXT_SOURCE_TYPES = {"text", "chunk", "llm_text"}
VISION_SOURCE_TYPES = {"vision", "cv", "computer_vision"}
SOURCE_PRIORITY = {
    "docling_table": 30,
    "camelot_table": 30,
    "table": 30,
    "vision": 25,
    "text": 20,
    "chunk": 20,
    "llm_text": 20,
    "unknown": 10,
}
OUTPUT_COLUMNS = [
    "normalized_formula",
    "material_formula",
    "material_name",
    "property_name",
    "value",
    "unit",
    "assay",
    "condition",
    "material_id",
    "source_type",
    "article_id",
    "source_id",
    "chunk_index",
    "quality_score",
    "quality_flags",
    "evidence",
]
CONFLICT_COLUMNS = [
    "normalized_formula",
    "property_name",
    "unit",
    "assay",
    "condition",
    "chosen_value",
    "chosen_source_type",
    "rejected_value",
    "rejected_source_type",
    "resolution",
    "article_id",
    "source_id",
]
REJECTED_COLUMNS = [
    "reason",
    "validator",
    "material_formula",
    "normalized_formula",
    "material_name",
    "property_name",
    "value",
    "unit",
    "assay",
    "condition",
    "material_id",
    "source_type",
    "article_id",
    "source_id",
    "chunk_index",
    "evidence",
]


@dataclass
class NanoAggregationResult:
    clean: pd.DataFrame
    conflicts: pd.DataFrame
    rejected: pd.DataFrame

    def write_csv(
        self,
        output_csv: str | Path,
        conflicts_csv: str | Path | None = None,
        rejected_csv: str | Path | None = None,
    ) -> None:
        output_path = Path(output_csv)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        self.clean.to_csv(output_path, index=False)

        if conflicts_csv:
            conflicts_path = Path(conflicts_csv)
            conflicts_path.parent.mkdir(parents=True, exist_ok=True)
            self.conflicts.to_csv(conflicts_path, index=False)

        if rejected_csv:
            rejected_path = Path(rejected_csv)
            rejected_path.parent.mkdir(parents=True, exist_ok=True)
            self.rejected.to_csv(rejected_path, index=False)


def aggregate_nanozyme_rows(
    rows: Iterable[dict[str, Any]],
    value_decimals: int = 6,
    conflict_tolerance: float = 1e-9,
) -> NanoAggregationResult:
    normalized_rows, rejected_rows = normalize_nano_rows(rows, value_decimals=value_decimals)
    if not normalized_rows:
        return NanoAggregationResult(
            clean=pd.DataFrame(columns=OUTPUT_COLUMNS),
            conflicts=pd.DataFrame(columns=CONFLICT_COLUMNS),
            rejected=pd.DataFrame(rejected_rows, columns=REJECTED_COLUMNS),
        )

    frame = pd.DataFrame(normalized_rows)
    frame = frame.sort_values(
        by=[
            "normalized_formula",
            "property_name",
            "unit",
            "assay",
            "condition",
            "source_priority",
            "_row_order",
        ],
        ascending=[True, True, True, True, True, False, True],
        kind="mergesort",
    )

    selected_rows: list[dict[str, Any]] = []
    conflict_rows: list[dict[str, Any]] = []
    group_columns = ["normalized_formula", "property_name", "unit", "assay", "condition"]
    for key, group in frame.groupby(group_columns, sort=True):
        selected = group.iloc[0].to_dict()
        selected_rows.append(selected)
        conflict_rows.extend(
            build_conflict_rows(
                key=dict(zip(group_columns, key)),
                selected=selected,
                alternatives=group.iloc[1:].to_dict("records"),
                conflict_tolerance=conflict_tolerance,
            )
        )

    clean = pd.DataFrame(selected_rows)
    clean = clean[OUTPUT_COLUMNS]
    conflicts = pd.DataFrame(conflict_rows, columns=CONFLICT_COLUMNS)
    rejected = pd.DataFrame(rejected_rows, columns=REJECTED_COLUMNS)
    return NanoAggregationResult(
        clean=clean.reset_index(drop=True),
        conflicts=conflicts.reset_index(drop=True),
        rejected=rejected.reset_index(drop=True),
    )


def normalize_nano_rows(rows: Iterable[dict[str, Any]], value_decimals: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    normalized: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for row_order, row in enumerate(rows):
        material_formula = normalize_text(row.get("material_formula") or row.get("formula"))
        normalized_formula = normalize_text(row.get("normalized_formula")) or normalize_formula(material_formula)
        property_name = normalize_text(row.get("property_name") or row.get("property"))
        unit = normalize_text(row.get("unit") or row.get("units"))
        value = normalize_numeric_value(row.get("value"), value_decimals=value_decimals)
        if not normalized_formula or not property_name or not unit or value is None:
            rejected.append(build_rejected_row(row, "missing_or_invalid_required_field", validator="nano_aggregation"))
            continue

        source_type = normalize_source_type(row.get("source_type"))
        normalized.append(
            add_quality_columns(
                {
                    "normalized_formula": normalized_formula,
                    "material_formula": material_formula or normalized_formula,
                    "material_name": normalize_text(row.get("material_name") or row.get("material")),
                    "property_name": property_name,
                    "value": value,
                    "unit": unit,
                    "assay": normalize_text(row.get("assay")),
                    "condition": normalize_text(row.get("condition")),
                    "material_id": normalize_text(row.get("material_id") or row.get("compound_id")),
                    "source_type": source_type,
                    "source_priority": SOURCE_PRIORITY[source_type],
                    "article_id": normalize_text(row.get("article_id")),
                    "source_id": normalize_text(row.get("source_id")),
                    "chunk_index": normalize_text(row.get("chunk_index")),
                    "evidence": normalize_text(row.get("evidence")),
                    "_row_order": row_order,
                },
                identifier_field="normalized_formula",
                identifier_flag="valid_formula",
            )
        )
    return normalized, rejected


def build_conflict_rows(
    key: dict[str, str],
    selected: dict[str, Any],
    alternatives: list[dict[str, Any]],
    conflict_tolerance: float,
) -> list[dict[str, Any]]:
    conflicts: list[dict[str, Any]] = []
    for alternative in alternatives:
        if values_equal(selected["value"], alternative["value"], tolerance=conflict_tolerance):
            continue
        conflicts.append(
            {
                **key,
                "chosen_value": selected["value"],
                "chosen_source_type": selected["source_type"],
                "rejected_value": alternative["value"],
                "rejected_source_type": alternative["source_type"],
                "resolution": conflict_resolution(selected, alternative),
                "article_id": alternative.get("article_id", ""),
                "source_id": alternative.get("source_id", ""),
            }
        )
    return conflicts


def conflict_resolution(selected: dict[str, Any], alternative: dict[str, Any]) -> str:
    if selected["source_priority"] > alternative["source_priority"]:
        return "table_priority" if selected["source_type"] in TABLE_SOURCE_TYPES else "source_priority"
    if selected["source_priority"] == alternative["source_priority"]:
        return "first_seen_same_priority"
    return "lower_priority_selected"


def normalize_formula(value: str) -> str:
    validation = validate_material_formula(value)
    return "" if validation.errors else validation.normalized_formula


def normalize_source_type(value: Any) -> str:
    source_type = normalize_text(value).lower() or "unknown"
    if source_type in SOURCE_PRIORITY:
        return source_type
    if source_type in TABLE_SOURCE_TYPES:
        return "table"
    if source_type in TEXT_SOURCE_TYPES:
        return "text"
    if source_type in VISION_SOURCE_TYPES:
        return "vision"
    return "unknown"


def normalize_numeric_value(value: Any, value_decimals: int) -> float | None:
    if value is None:
        return None
    text = normalize_text(value).replace(",", ".")
    if not text:
        return None
    try:
        number = float(text)
    except ValueError:
        return None
    if not math.isfinite(number):
        return None
    return round(number, value_decimals)


def values_equal(left: float, right: float, tolerance: float) -> bool:
    return abs(float(left) - float(right)) <= tolerance


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def build_rejected_row(row: dict[str, Any], reason: str, validator: str) -> dict[str, Any]:
    material_formula = normalize_text(row.get("material_formula") or row.get("formula"))
    return {
        "reason": normalize_text(row.get("reason")) or reason,
        "validator": normalize_text(row.get("validator")) or validator,
        "material_formula": material_formula,
        "normalized_formula": normalize_text(row.get("normalized_formula")) or normalize_formula(material_formula),
        "material_name": normalize_text(row.get("material_name") or row.get("material")),
        "property_name": normalize_text(row.get("property_name") or row.get("property")),
        "value": normalize_text(row.get("value")),
        "unit": normalize_text(row.get("unit") or row.get("units")),
        "assay": normalize_text(row.get("assay")),
        "condition": normalize_text(row.get("condition")),
        "material_id": normalize_text(row.get("material_id") or row.get("compound_id")),
        "source_type": normalize_text(row.get("source_type")),
        "article_id": normalize_text(row.get("article_id")),
        "source_id": normalize_text(row.get("source_id")),
        "chunk_index": normalize_text(row.get("chunk_index")),
        "evidence": normalize_text(row.get("evidence")),
    }
