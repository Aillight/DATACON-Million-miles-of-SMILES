from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


TABLE_SOURCE_TYPES = {"table", "docling_table", "camelot_table"}
TEXT_SOURCE_TYPES = {"text", "chunk", "llm_text"}
SOURCE_PRIORITY = {
    "docling_table": 30,
    "camelot_table": 30,
    "table": 30,
    "text": 20,
    "chunk": 20,
    "llm_text": 20,
    "unknown": 10,
}
OUTPUT_COLUMNS = [
    "canonical_smiles",
    "property_name",
    "value",
    "unit",
    "compound_id",
    "smiles",
    "source_type",
    "article_id",
    "source_id",
    "chunk_index",
    "evidence",
]
CONFLICT_COLUMNS = [
    "canonical_smiles",
    "property_name",
    "chosen_value",
    "chosen_source_type",
    "rejected_value",
    "rejected_source_type",
    "resolution",
    "article_id",
    "source_id",
]


@dataclass
class AggregationResult:
    clean: pd.DataFrame
    conflicts: pd.DataFrame

    def write_csv(self, output_csv: str | Path, conflicts_csv: str | Path | None = None) -> None:
        output_path = Path(output_csv)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        self.clean.to_csv(output_path, index=False)

        if conflicts_csv:
            conflicts_path = Path(conflicts_csv)
            conflicts_path.parent.mkdir(parents=True, exist_ok=True)
            self.conflicts.to_csv(conflicts_path, index=False)


def aggregate_extraction_rows(
    rows: Iterable[dict[str, Any]],
    value_decimals: int = 3,
    conflict_tolerance: float = 1e-9,
) -> AggregationResult:
    normalized_rows = normalize_rows(rows, value_decimals=value_decimals)
    if not normalized_rows:
        return AggregationResult(
            clean=pd.DataFrame(columns=OUTPUT_COLUMNS),
            conflicts=pd.DataFrame(columns=CONFLICT_COLUMNS),
        )

    frame = pd.DataFrame(normalized_rows)
    frame = frame.sort_values(
        by=["canonical_smiles", "property_name", "source_priority", "_row_order"],
        ascending=[True, True, False, True],
        kind="mergesort",
    )

    selected_rows: list[dict[str, Any]] = []
    conflict_rows: list[dict[str, Any]] = []
    for (canonical_smiles, property_name), group in frame.groupby(["canonical_smiles", "property_name"], sort=True):
        selected = group.iloc[0].to_dict()
        selected_rows.append(selected)
        conflict_rows.extend(
            build_conflict_rows(
                canonical_smiles=str(canonical_smiles),
                property_name=str(property_name),
                selected=selected,
                alternatives=group.iloc[1:].to_dict("records"),
                conflict_tolerance=conflict_tolerance,
            )
        )

    clean = pd.DataFrame(selected_rows)
    clean = clean[OUTPUT_COLUMNS]
    conflicts = pd.DataFrame(conflict_rows, columns=CONFLICT_COLUMNS)
    return AggregationResult(clean=clean.reset_index(drop=True), conflicts=conflicts.reset_index(drop=True))


def normalize_rows(rows: Iterable[dict[str, Any]], value_decimals: int) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for row_order, row in enumerate(rows):
        canonical_smiles = normalize_text(row.get("canonical_smiles")) or canonicalize_smiles(row.get("smiles"))
        property_name = normalize_text(row.get("property_name") or row.get("property") or "pMIC")
        value = normalize_numeric_value(row.get("value"), value_decimals=value_decimals)
        if not canonical_smiles or not property_name or value is None:
            continue

        source_type = normalize_source_type(row.get("source_type"))
        normalized.append(
            {
                "canonical_smiles": canonical_smiles,
                "property_name": property_name,
                "value": value,
                "unit": normalize_text(row.get("unit")),
                "compound_id": normalize_text(row.get("compound_id")),
                "smiles": normalize_text(row.get("smiles")),
                "source_type": source_type,
                "source_priority": SOURCE_PRIORITY[source_type],
                "article_id": normalize_text(row.get("article_id")),
                "source_id": normalize_text(row.get("source_id")),
                "chunk_index": normalize_text(row.get("chunk_index")),
                "evidence": normalize_text(row.get("evidence")),
                "_row_order": row_order,
            }
        )
    return normalized


def build_conflict_rows(
    canonical_smiles: str,
    property_name: str,
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
                "canonical_smiles": canonical_smiles,
                "property_name": property_name,
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


def values_equal(left: float, right: float, tolerance: float) -> bool:
    return abs(float(left) - float(right)) <= tolerance


def normalize_source_type(value: Any) -> str:
    source_type = normalize_text(value).lower() or "unknown"
    if source_type in SOURCE_PRIORITY:
        return source_type
    if source_type in TABLE_SOURCE_TYPES:
        return "table"
    if source_type in TEXT_SOURCE_TYPES:
        return "text"
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


def canonicalize_smiles(value: Any) -> str:
    smiles = normalize_text(value)
    if not smiles:
        return ""
    try:
        from rdkit import Chem
        from rdkit import RDLogger
    except ImportError:
        return smiles

    RDLogger.DisableLog("rdApp.error")
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        return ""
    return Chem.MolToSmiles(molecule, canonical=True)


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()
