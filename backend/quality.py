from __future__ import annotations

from typing import Any


TABLE_SOURCE_TYPES = {"table", "docling_table", "camelot_table"}
TEXT_SOURCE_TYPES = {"text", "chunk", "llm_text"}
VISION_SOURCE_TYPES = {"vision", "cv", "computer_vision"}


def add_quality_columns(
    row: dict[str, Any],
    *,
    identifier_field: str,
    identifier_flag: str,
) -> dict[str, Any]:
    flags: list[str] = []
    score = 0

    source_type = normalize_text(row.get("source_type")).lower()
    if source_type in TABLE_SOURCE_TYPES:
        flags.append("from_table")
        score += 30
    elif source_type in VISION_SOURCE_TYPES:
        flags.append("from_vision")
        score += 25
    elif source_type in TEXT_SOURCE_TYPES:
        flags.append("from_text")
        score += 20
    else:
        flags.append("source_unknown")
        score += 10

    if normalize_text(row.get(identifier_field)):
        flags.append(identifier_flag)
        score += 25
    else:
        flags.append("missing_identifier")

    if normalize_text(row.get("unit")):
        flags.append("has_unit")
        score += 15
    else:
        flags.append("missing_unit")

    if normalize_text(row.get("evidence")):
        flags.append("has_evidence")
        score += 20
    else:
        flags.append("missing_evidence")

    if normalize_text(row.get("condition")):
        flags.append("has_condition")
        score += 5

    enriched = dict(row)
    enriched["quality_score"] = min(score, 100)
    enriched["quality_flags"] = ";".join(flags)
    return enriched


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()
