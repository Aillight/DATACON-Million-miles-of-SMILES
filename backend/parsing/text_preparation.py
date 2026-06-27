from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from backend.parsing.chunking import TextChunk, chunk_article_text
from backend.parsing.pdf_parser import parse_pdf_to_markdown


TRAILING_SECTION_RE = re.compile(
    r"(?im)^\s*(?:#{1,6}\s*)?(?:\d+(?:\.\d+)*\.?\s*)?"
    r"(references?|bibliography|literature cited|references and notes|"
    r"acknowledg(?:e)?ments?|funding)\s*$"
)
APPENDED_TABLE_RE = re.compile(r"(?m)^## Table \d+(?:\s*\([^)]*\))?\s*$")
PAGE_NUMBER_RE = re.compile(r"(?im)^\s*(?:page\s*)?\d{1,4}\s*(?:of\s+\d{1,4})?\s*$")
PAGE_ARTIFACT_RE = re.compile(
    r"(?im)^\s*(?:downloaded from .+|"
    r"(?:https?://|www\.).+|"
    r"doi:\s*10\.\S+|"
    r"(?:copyright|\u00a9).+|"
    r"all rights reserved\.?)\s*$"
)
MICROGRAM_UNIT_RE = r"(?:ug|mcg|\u00b5g|\u03bcg|microgram(?:s)?)\s*(?:/|per)\s*m(?:l|L)"
MICROGRAM_PER_ML_RE = re.compile(
    rf"(?P<relation><=|>=|<|>|=|~|ca\.?|about|approximately)?\s*"
    rf"(?P<value>\d+(?:[.,]\d+)?)"
    rf"(?:\s*(?:-|\u2013|\u2014|to)\s*(?P<upper_value>\d+(?:[.,]\d+)?))?"
    rf"\s*(?P<unit>{MICROGRAM_UNIT_RE})",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class ConcentrationMention:
    raw_text: str
    value_ug_ml: float
    upper_value_ug_ml: float | None = None
    pmic: float | None = None
    upper_pmic: float | None = None
    relation: str | None = None
    start_char: int = 0
    end_char: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PreparedDocument:
    source: str
    parser: str
    markdown: str
    chunks: list[TextChunk]
    concentration_mentions: list[ConcentrationMention] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["chunks"] = [chunk.to_dict() for chunk in self.chunks]
        payload["concentration_mentions"] = [mention.to_dict() for mention in self.concentration_mentions]
        return payload


def prepare_pdf_for_extraction(
    pdf_path: str | Path,
    target_sections: list[str] | tuple[str, ...] | None = None,
    max_chars: int = 6000,
    overlap_chars: int = 400,
    concentration_smiles: str | None = None,
) -> PreparedDocument:
    parsed = parse_pdf_to_markdown(pdf_path)
    return prepare_markdown_for_extraction(
        markdown=parsed.markdown,
        source=parsed.source,
        parser=parsed.parser,
        warnings=parsed.warnings,
        target_sections=target_sections,
        max_chars=max_chars,
        overlap_chars=overlap_chars,
        concentration_smiles=concentration_smiles,
    )


def prepare_markdown_for_extraction(
    markdown: str,
    source: str = "document",
    parser: str = "markdown",
    warnings: list[str] | None = None,
    target_sections: list[str] | tuple[str, ...] | None = None,
    max_chars: int = 6000,
    overlap_chars: int = 400,
    concentration_smiles: str | None = None,
) -> PreparedDocument:
    cleaned_markdown = clean_parsed_markdown(markdown)
    chunks = chunk_article_text(
        text=cleaned_markdown,
        target_sections=target_sections,
        max_chars=max_chars,
        overlap_chars=overlap_chars,
    )
    mentions = find_microgram_per_ml_mentions(cleaned_markdown, smiles=concentration_smiles)
    return PreparedDocument(
        source=source,
        parser=parser,
        markdown=cleaned_markdown,
        chunks=chunks,
        concentration_mentions=mentions,
        warnings=warnings or [],
    )


def clean_parsed_markdown(markdown: str) -> str:
    match = APPENDED_TABLE_RE.search(markdown)
    if not match:
        return clean_article_markdown(markdown)

    body = clean_article_markdown(markdown[: match.start()])
    tables = markdown[match.start() :].strip()
    return "\n\n".join(part for part in (body, tables) if part)


def clean_article_markdown(markdown: str) -> str:
    text = normalize_line_endings(markdown)
    text = cut_after_trailing_sections(text)
    text = remove_page_artifacts(text)
    text = re.sub(r"[ \t]+$", "", text, flags=re.MULTILINE)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def cut_after_trailing_sections(text: str) -> str:
    match = TRAILING_SECTION_RE.search(text)
    if not match:
        return text
    return text[: match.start()].rstrip()


def remove_page_artifacts(text: str) -> str:
    text = PAGE_NUMBER_RE.sub("", text)
    text = PAGE_ARTIFACT_RE.sub("", text)

    lines = text.split("\n")
    repeated = find_repeated_artifact_lines(lines)
    if not repeated:
        return "\n".join(lines)

    return "\n".join(line for line in lines if normalize_artifact_line(line) not in repeated)


def find_repeated_artifact_lines(lines: list[str]) -> set[str]:
    normalized_lines = [normalize_artifact_line(line) for line in lines]
    counts = Counter(line for line in normalized_lines if line)
    repeated: set[str] = set()
    for raw_line, normalized in zip(lines, normalized_lines):
        if counts[normalized] < 3:
            continue
        if looks_like_repeated_page_artifact(raw_line):
            repeated.add(normalized)
    return repeated


def looks_like_repeated_page_artifact(line: str) -> bool:
    text = line.strip().lower()
    if not text or len(text) > 120 or text.startswith("|"):
        return False
    artifact_tokens = (
        "downloaded",
        "copyright",
        "all rights reserved",
        "doi:",
        "journal",
        "volume",
        "vol.",
        "issue",
        "www.",
        "http://",
        "https://",
    )
    return any(token in text for token in artifact_tokens)


def find_microgram_per_ml_mentions(text: str, smiles: str | None = None) -> list[ConcentrationMention]:
    molecular_weight = molecular_weight_from_smiles(smiles) if smiles else None
    mentions: list[ConcentrationMention] = []
    for match in MICROGRAM_PER_ML_RE.finditer(text):
        value = parse_float(match.group("value"))
        if value is None:
            continue
        upper_value = parse_float(match.group("upper_value"))
        mentions.append(
            ConcentrationMention(
                raw_text=match.group(0).strip(),
                relation=normalize_relation(match.group("relation")),
                value_ug_ml=value,
                upper_value_ug_ml=upper_value,
                pmic=ug_ml_to_pmic(value, molecular_weight),
                upper_pmic=ug_ml_to_pmic(upper_value, molecular_weight) if upper_value is not None else None,
                start_char=match.start(),
                end_char=match.end(),
            )
        )
    return mentions


def ug_ml_to_pmic(value_ug_ml: float | None, molecular_weight: float | None) -> float | None:
    if value_ug_ml is None or molecular_weight is None or molecular_weight <= 0 or value_ug_ml <= 0:
        return None
    molar = (value_ug_ml * 1e-3) / molecular_weight
    return -math.log10(molar)


def molecular_weight_from_smiles(smiles: str | None) -> float | None:
    if not smiles:
        return None
    try:
        from rdkit import Chem
        from rdkit.Chem import Descriptors
    except ImportError:
        return None

    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        return None
    return float(Descriptors.MolWt(molecule))


def parse_float(value: str | None) -> float | None:
    if not value:
        return None
    try:
        return float(value.replace(",", "."))
    except ValueError:
        return None


def normalize_relation(value: str | None) -> str | None:
    if not value:
        return None
    normalized = value.strip().lower()
    if normalized in {"ca.", "about", "approximately"}:
        return "~"
    return normalized


def normalize_line_endings(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n")


def normalize_artifact_line(line: str) -> str:
    return re.sub(r"\s+", " ", line.strip().lower())
